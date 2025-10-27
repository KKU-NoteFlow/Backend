from __future__ import annotations
import re, asyncio, os, threading, json, time
from typing import Optional, List

# LangChain / Ollama (옵션)
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# HF Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# 웹 보강(옵션)
import requests
from urllib.parse import quote as _urlquote


# =========================================================
# 설정값 (환경변수로 오버라이드 가능)
# =========================================================
DEFAULT_SUMMARY_BACKEND = os.getenv("SUMMARY_BACKEND", "hf").lower()  # "hf" | "ollama"
DEFAULT_LONGDOC_CHAR_LIMIT = int(os.getenv("SUMMARY_LONGDOC_CHAR_LIMIT", "3500"))

# 롱독 청크 설정
DEFAULT_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "12000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "1200"))

# 토큰 예산
HF_MAX_NEW_TOKENS_SHORT = int(os.getenv("HF_MAX_NEW_TOKENS_SHORT", "8000"))
HF_MAX_NEW_TOKENS_MEDIUM = int(os.getenv("HF_MAX_NEW_TOKENS_MEDIUM", "16000"))
HF_MAX_NEW_TOKENS_LONG = int(os.getenv("HF_MAX_NEW_TOKENS_LONG", "32000"))
HF_MAP_MAX_NEW_TOKENS = int(os.getenv("HF_MAP_MAX_NEW_TOKENS", "12000"))

# 슬라이드 커버리지 목표
SLIDES_MIN = int(os.getenv("SUMMARY_SLIDES_MIN", "8"))
SLIDES_MAX = int(os.getenv("SUMMARY_SLIDES_MAX", "40"))

# 증분 보강 루프 횟수
ENSURE_COMPLETION_PASSES = int(os.getenv("ENSURE_COMPLETION_PASSES", "3"))


# =========================================================
# 사고과정 유사 문장 필터 (stream시 메타 프레이즈 억제)
# =========================================================
_THOUGHT_PAT = re.compile(
    r"^\s*(okay|let\s+me|i\s+need\s+to|in summary)\b",
    re.I,
)

# =========================================================
# 마크다운 보정 유틸
# =========================================================
_MD_CODE_FENCE = re.compile(r"(```.*?```|`[^`]*`)", re.S)

def _format_md_stream(s: str) -> str:
    if not s:
        return s
    parts = []
    last = 0
    for m in _MD_CODE_FENCE.finditer(s):
        chunk = s[last:m.start()]
        parts.append(_format_md_plain(chunk))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_format_md_plain(s[last:]))
    out = "".join(parts)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out

def _format_md_plain(s: str) -> str:
    s = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", s, flags=re.M)
    s = re.sub(r"(?<!\n)\s*(#{1,6}\s)", r"\n\n\1", s)
    s = re.sub(r"(?m)(^#{1,6}\s.*$)", r"\n\1\n", s)
    s = re.sub(r"(?<!\n)\s*(-\s+)", r"\n\1", s)
    s = re.sub(r"(?<!\n)\s*(\d+\.\s+)", r"\n\1", s)
    s = re.sub(r"(\n[-\d].*\S)(?=[^\n]|$)", r"\1", s)
    return s


# =========================================================
# 도메인/언어 감지
# =========================================================
def _detect_domain(t: str) -> str:
    s = (t or "").lower()
    if re.search(r"\blecture\b|강의|슬라이드|ppt|slide|강의자료|강의록", s):
        return "lecture"
    if re.search(r"\b(def |class |import |#include|public\s+class|function\s|=>|:=)", s) or re.search(r"```|\bdiff --git\b|\bcommit\b", s):
        return "code"
    if re.search(r"\babstract\b|\bintroduction\b|\bmethod(s)?\b|\bresult(s)?\b|\bconclusion(s)?\b|doi:|arxiv:\d", s):
        return "paper"
    if re.search(r"회의|안건|결정|논의|액션 아이템|참석자|회의록|meeting|agenda|minutes|action items|attendees", s):
        return "meeting"
    return "general"

def _detect_lang(t: str) -> str:
    s = t or ""
    en = len(re.findall(r"[A-Za-z]", s))
    ko = len(re.findall(r"[가-힣]", s))
    return "en" if en > ko else "ko"


# =========================================================
# 시스템 프롬프트 (슬라이드 섹션 강제)
# =========================================================
def _system_prompt(domain: str, phase: str = "final", output_format: str = "md", length: str = "medium") -> str:
    fmt = output_format.lower()
    base_rules = (
        "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 텍스트의 언어(Korean/English)를 감지하고, 반드시 동일한 언어로 작성한다. "
        "요약의 목적은 읽기 쉬운, GPT 스타일의 명확한 결과물을 만드는 것이다. 핵심 정보(주요 주장/결론, 인물·기관·수치·날짜·지표·범위, 원인↔결과·조건·한계)를 빠짐없이 담되 군더더기와 반복을 제거한다. "
        "추정·가치판단·조언은 입력에서 명시적 근거가 없으면 추가하지 말고, 사고과정(Chain-of-Thought)이나 중간 추론은 출력하지 마라."
    )
    if domain == "meeting":
        include_hint = "결정 사항, 책임자/기한이 명시된 액션, 남은 이슈, 다음 단계"
    elif domain == "lecture":
        include_hint = "핵심 개념, 주요 정의/공식, 예제/응용, 학습 포인트(요약된 학습 목표), 참고 자료"
    elif domain == "code":
        include_hint = "변경 목적/범위, 주요 API/모듈 영향, 호환성/리스크, 마이그레이션 포인트"
    elif domain == "paper":
        include_hint = "문제 정의·동기, 방법 요지, 데이터/세팅, 핵심 수치, 한계/가정"
    else:
        include_hint = "5W1H, 핵심 주장/결과, 중요한 수치·날짜·고유명사, 원인↔결과·조건, 한계/주의점"

    if fmt == "md":
        format_rule = (
            "출력 형식: Markdown. 반드시 다음 섹션으로 구성하라(필요시 일부 생략 가능): "
            "## TL;DR, ## 핵심 요점(불릿 3–8개), ## 상세 설명(문단), ## 슬라이드(필수), ## 용어 정리(선택), ## 한계/주의, ## 할 일(액션), ## 참고(선택). "
            "절대 H1('# ')로 시작하지 말고, 불필요한 전언/사고과정/추론 과정을 출력하지 마라."
        )
    else:
        format_rule = (
            "출력 형식: HTML fragment. <h1>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>만 사용. "
            "<h1>제목</h1>, <h2>개요</h2>, <h2>핵심 요점</h2><ul>…</ul>, <h2>상세 설명</h2>, "
            "<h2>슬라이드</h2>, <h2>용어 정리</h2>, <h2>한계/주의</h2>, <h2>할 일</h2>, <h2>참고/추가자료</h2>의 순서."
        )

    if length == 'long':
        length_rule = "분량: 충분히 상세하게, 원문 전반의 주요 포인트·예시·숫자·결론을 모두 포함하라(원문 대비 30–70% 분량 권장, 또는 토큰 예산 한도 내 최대로)."
    elif length == 'short':
        length_rule = "분량: 한두 문장 TL;DR 중심(간결)."
    else:
        length_rule = "분량: 원문 대비 약 15–30%. 각 문단은 2–5문장."

    md_spacing_rule = (
        "마크다운 간격 규칙: 모든 헤더(#, ##, ### 등) 뒤에는 한 칸 공백을 두고, 헤더의 앞뒤에는 빈 줄 1줄을 둔다. "
        "불릿(- )은 항목마다 줄바꿈하고, 서브항목은 들여쓰기 2–4칸을 사용한다."
    )
    web_rule = (
        "추가자료가 주어지면 원문과 교차 검증하여 모순이 있으면 원문을 우선하라. "
        "추가자료의 출처명(예: 위키백과)을 괄호로 간단히 표기할 수 있다."
    )

    if phase == "map":
        scope = (
            "이 청크만 대상으로 섹션 골격을 간략히 채워라. 특히 **## 슬라이드** 섹션에 이 청크의 주요 하위 주제를 1–3장의 "
            "‘### 슬라이드 n: 제목’ + 불릿(3–6개)로 만들어라. 슬라이드 번호는 임시로 두고, 리듀스 단계에서 재번호됨."
        )
    elif phase == "reduce":
        scope = (
            "아래 청크 요약들을 중복 없이 통합해 일관된 섹션 구성을 완성하라. 흐름(원인→과정→결과)을 유지. "
            "특히 **## 슬라이드** 섹션에 모든 청크의 슬라이드를 병합·정리하여 누락 없이 포함하라. "
            "슬라이드 번호는 1부터 순차 재배열하고, 최소 목표 슬라이드 수(<!-- target_slides:X --> 주어짐)를 충족하라."
        )
    else:
        scope = (
            "전체 텍스트를 위 섹션 구조에 맞춰 응집력 있게 작성하라. 출력은 반드시 Markdown만 사용하라(원시 HTML 금지). "
            "최상단 제목(H1)은 생략하고, '## TL;DR'로 시작하라. "
            "특히 **## 슬라이드** 섹션을 포함하고, 슬라이드를 ‘### 슬라이드 1: …’ 형식으로 최소 목표 수(<!-- target_slides:X -->) 이상 생성하라. "
            "각 슬라이드는 3–6개 불릿을 갖고, 제목은 중복되지 않게 만든다."
        )

    example = (
        "\n\n--- 예시 출력 (한국어, medium) ---\n"
        "## TL;DR\n"
        "프로젝트 A의 기능 X가 2주 지연되어 배포 일정이 11/10로 변경됨. 주요 리스크는 외부 API 응답 지연.\n\n"
        "## 핵심 요점\n"
        "- 기능 X 구현 지연: 2주\n"
        "- 배포 일정: 11/10\n"
        "- QA 담당: 민수\n\n"
        "## 슬라이드\n"
        "### 슬라이드 1: 일정 변경 배경\n"
        "- 외부 API 응답 지연이 주요 원인\n"
        "- 기능 X 의존성이 높음\n"
        "- …\n"
        "### 슬라이드 2: 리스크와 대응\n"
        "- 타임아웃 상향 및 캐시\n"
        "- …\n\n"
        "## 할 일\n"
        "- [개발팀] API 응답 문제 원인 분석 — 11/01\n"
        "-------------------------------\n\n"
    )

    meta_hint = (
        "\n\n출력 후 반드시 JSON 메타데이터 블록을 추가하라. "
        "형식: ```json\n{ \"tl_dr\": \"...\", \"tags\": [\"t1\",\"t2\"], \"actions\": [{\"assignee\": \"name\", \"task\": \"...\", \"due\": \"YYYY-MM-DD\"}], \"language\": \"ko\", \"missing\": [] }\n```\n"
    )

    return f"{base_rules}\n포함 우선: {include_hint}\n{format_rule}\n{length_rule}\n{md_spacing_rule}\n{web_rule}\n{scope}{example}{meta_hint}"


# =========================================================
# 웹 보강 유틸 (옵션)
# =========================================================
def _is_augmentation_allowed() -> bool:
    return os.getenv("AUGMENT_WEB", "false").lower() in ("1", "true", "yes")

def _extract_entities_for_web(text: str, lang: str = "ko", max_items: int = 5) -> list[str]:
    items: list[str] = []
    if lang == "en":
        items = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    else:
        items = re.findall(r"[「『“\"'(](.*?)[」』”\"')]", text) + re.findall(r"[가-힣A-Za-z][가-힣A-Za-z]{1,}", text)
    stop = set(["회의", "안건", "결정", "논의", "데이터", "모델", "결과", "방법", "프로젝트", "사용자", "시스템"])
    uniq = []
    for w in items:
        w = re.sub(r"[\s]+", " ", w).strip()
        if 2 <= len(w) <= 80 and w not in stop and not re.match(r"^\d+$", w):
            if w not in uniq:
                uniq.append(w)
    return uniq[:max_items]

def _fetch_wikipedia_summaries(entities: list[str], lang: str = "ko", max_sources: int = 5) -> str:
    base = "https://%s.wikipedia.org/api/rest_v1/page/summary/%s" % (("ko" if lang == "ko" else "en"), "%s")
    out = []
    timeout = float(os.getenv("AUGMENT_HTTP_TIMEOUT", "2.5"))
    session = requests.Session()
    headers = {"User-Agent": os.getenv("AUGMENT_UA", "SummaryAgent/1.0")}
    for ent in entities[:max_sources]:
        url = base % _urlquote(ent)
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            data = r.json()
            title = data.get("title") or ent
            extract = (data.get("extract") or "").strip()
            if not extract:
                continue
            extract = (extract[:500] + "…") if len(extract) > 500 else extract
            src = "위키백과" if lang == "ko" else "Wikipedia"
            out.append(f"- **{title}** ({src}): {extract}")
            time.sleep(0.05)
        except Exception:
            continue
    return "\n".join(out)


# =========================================================
# 프롬프트 빌드 / HF 모델 로딩
# =========================================================
_HF_MODEL = None
_HF_TOKENIZER = None
_HF_NAME = None

def _resolve_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None

def _load_hf_model():
    global _HF_MODEL, _HF_TOKENIZER, _HF_NAME
    if _HF_MODEL is not None and _HF_TOKENIZER is not None:
        return _HF_MODEL, _HF_TOKENIZER

    primary = os.getenv("HF_SUMMARY_MODEL", "CohereForAI/aya-23-8B")
    fallback = os.getenv("HF_SUMMARY_FALLBACK_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    token = os.getenv("HF_API_TOKEN") or None
    torch_dtype = _resolve_dtype()
    load_in_4bit = os.getenv("HF_LOAD_IN_4BIT", "false").lower() in ("1", "true", "yes")

    def try_load(name: str):
        hub_kwargs = {"trust_remote_code": True}
        if token:
            hub_kwargs["token"] = token
        try:
            tok = AutoTokenizer.from_pretrained(name, **hub_kwargs)
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                hk = dict(hub_kwargs); hk.pop("token", None)
                tok = AutoTokenizer.from_pretrained(name, **hk)
            else:
                raise
        kwargs = dict(hub_kwargs)
        if load_in_4bit:
            try:
                kwargs.update({"load_in_4bit": True})
            except Exception:
                pass
        else:
            if torch_dtype is not None:
                kwargs["dtype"] = torch_dtype

        def load_model(with_token: bool):
            mk = dict(kwargs)
            if not with_token and "token" in mk:
                mk.pop("token", None)
            return AutoModelForCausalLM.from_pretrained(name, device_map="auto", **mk)

        try:
            model = load_model(with_token=bool(token))
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                model = load_model(with_token=False)
            else:
                try:
                    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                except Exception:
                    mk = dict(kwargs); mk.pop("token", None)
                    model = AutoModelForCausalLM.from_pretrained(name, **mk)
                if torch.cuda.is_available():
                    try: model.to("cuda")
                    except Exception: pass
        return model, tok

    try:
        model, tok = try_load(primary)
        _HF_MODEL, _HF_TOKENIZER, _HF_NAME = model, tok, primary
        return _HF_MODEL, _HF_TOKENIZER
    except Exception as e:
        if os.getenv("HF_DISABLE_FALLBACK", "1").lower() in ("1", "true", "yes"):
            raise RuntimeError("HF_DISABLED") from e
        try:
            model, tok = try_load(fallback)
            _HF_MODEL, _HF_TOKENIZER, _HF_NAME = model, tok, fallback
            return _HF_MODEL, _HF_TOKENIZER
        except Exception as e2:
            raise RuntimeError("HF_DISABLED") from e2


def _build_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            "You are a precise summarizer. Detect the input language (Korean/English) and write the summary in the SAME language. "
            "Preserve key facts; remove fluff; avoid speculation and chain-of-thought. "
            "Use Markdown sections: ## TL;DR, ## Key Points, ## Details, ## Slides, ## Terms, ## Limitations, ## Actions, ## References.\n\n"
            "Text:\n" + user_text + "\n\nSummary:"
        )


# =========================================================
# 합본 payload (슬라이드 목표 힌트 포함)
# =========================================================
def _compose_user_payload(
    main_text: str,
    extra_context: str,
    output_format: str,
    length: str = "medium",
    tone: str = "neutral",
    target_slides: Optional[int] = None
) -> str:
    fmt = output_format.lower()
    pref = f"요약 길이: {length}. 톤: {tone}."
    slide_hint = f"<!-- target_slides:{target_slides} -->" if target_slides else ""
    if fmt == "md":
        if extra_context:
            return (
                "## 원문\n"
                f"{main_text}\n\n"
                "## 추가자료(요약)\n"
                f"{extra_context}\n\n"
                f"<!-- 사용자 선호: {pref} -->\n"
                f"{slide_hint}\n"
            )
        return f"{main_text}\n\n<!-- 사용자 선호: {pref} -->\n{slide_hint}\n"
    else:
        if extra_context:
            return f"<h2>원문</h2>\n{main_text}\n\n<h2>추가자료(요약)</h2>\n{extra_context}\n<!-- 사용자 선호: {pref} -->\n{slide_hint}\n"
        return f"{main_text}\n<!-- 사용자 선호: {pref} -->\n{slide_hint}\n"


# =========================================================
# HF Streaming / Single-Shot
# =========================================================
def _simple_fallback_summary(text: str, output_format: str = "md") -> list[str]:
    s = (text or "").strip()
    if not s:
        return ["요약할 내용이 없습니다."]
    parts = re.split(r"(?<=[.!?。])\s+|\n+", s)
    parts = [p.strip() for p in parts if p.strip()]
    head = parts[:6]
    bullets = [f"- {p[:120]}{'…' if len(p) > 120 else ''}" for p in head[1:6]]
    if output_format.lower() == "md":
        out = [f"# 요약", head[0][:160] + ("…" if len(head[0]) > 160 else ""), "\n" ] + bullets
    else:
        out = [f"<h1>요약</h1>", f"<p>{head[0][:160]}{'…' if len(head[0])>160 else ''}</p>"] + [f"<li>{b[2:]}</li>" for b in bullets]
    return out

async def _stream_with_hf(text: str, system_text: str | None = None, output_format: str = "md"):
    try:
        model, tokenizer = _load_hf_model()
    except Exception:
        for line in _simple_fallback_summary(text, output_format=output_format):
            yield f"data: {line}\n\n"
        return

    sys_msg = system_text or (
        "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 언어를 감지하고 동일 언어로 작성한다. "
        "Markdown 섹션(## TL;DR, ## 핵심 요점, ## 상세 설명, ## 슬라이드, ## 용어 정리, ## 한계/주의, ## 할 일, ## 참고/추가자료)을 사용한다. "
        "추정/가치판단/사고과정 금지. 각 문단 2–5문장. 마크다운 간격 규칙을 지켜라."
    )
    prompt = _build_prompt(tokenizer, sys_msg, text)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        try:
            inputs = {k: v.to(getattr(model, "device", "cuda")) for k, v in inputs.items()}
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", str(HF_MAX_NEW_TOKENS_LONG))),
        do_sample=False,
        repetition_penalty=float(os.getenv("HF_REPETITION_PENALTY", "1.02")),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        streamer=streamer,
    )
    if gen_kwargs.get("do_sample"):
        gen_kwargs["temperature"] = float(os.getenv("HF_TEMPERATURE", "0.1"))

    def _gen():
        _ = model.generate(**inputs, **gen_kwargs)

    thread = threading.Thread(target=_gen, daemon=True)
    thread.start()

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _drain_streamer():
        for chunk in streamer:
            try:
                asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
            except Exception:
                break
        try:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)
        except Exception:
            pass

    threading.Thread(target=_drain_streamer, daemon=True).start()

    buffer = ""
    while True:
        chunk = await q.get()
        if chunk is None:
            break
        buffer += chunk
        if buffer.endswith(("\n", "。", ".", "…", "!", "?", ")", "]")):
            line = buffer
            buffer = ""
            if output_format.lower() == "md":
                line = _format_md_stream(line)
            if not _THOUGHT_PAT.match(line.strip()):
                yield f"data: {line}\n\n"

    if buffer.strip():
        line = buffer
        if output_format.lower() == "md":
            line = _format_md_stream(line)
        if not _THOUGHT_PAT.match(line.strip()):
            yield f"data: {line}\n\n"


async def _hf_generate_once(system_text: str, user_text: str, max_new_tokens: int = 256) -> str:
    model, tokenizer = _load_hf_model()
    prompt = _build_prompt(tokenizer, system_text, user_text)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        try:
            inputs = {k: v.to(getattr(model, "device", "cuda")) for k, v in inputs.items()}
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# =========================================================
# 텍스트 청크
# =========================================================
def _chunk_text(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text or ""
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# =========================================================
# Ollama Streaming (옵션)
# =========================================================
async def _stream_with_ollama(text: str, system_text: str | None = None, output_format: str = "md"):
    cb = AsyncIteratorCallbackHandler()

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    primary_model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    fallback_model = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:7b")
    num_ctx = os.getenv("OLLAMA_NUM_CTX")
    num_thread = os.getenv("OLLAMA_NUM_THREAD")

    def make_llm(model_name: str) -> ChatOllama:
        kwargs = {}
        if num_ctx:
            try:
                kwargs["num_ctx"] = int(num_ctx)
            except ValueError:
                pass
        if num_thread:
            try:
                kwargs["num_thread"] = int(num_thread)
            except ValueError:
                pass
        temp = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
        return ChatOllama(
            base_url=base_url,
            model=model_name,
            streaming=True,
            callbacks=[cb],
            temperature=temp,
            **kwargs,
        )

    llm = make_llm(primary_model)

    messages = [
        SystemMessage(
            content=(
                system_text or (
                    "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 언어를 감지하고 동일 언어로 작성한다. "
                    "Markdown 섹션(## TL;DR, ## 핵심 요점, ## 상세 설명, ## 슬라이드, ## 용어 정리, ## 한계/주의, ## 할 일, ## 참고/추가자료)을 사용한다. "
                    "추정/가치판단/사고과정 금지. 각 문단 2–5문장. 마크다운 간격 규칙을 지켜라."
                )
            )
        ),
        HumanMessage(content=text),
    ]

    try:
        task = asyncio.create_task(llm.agenerate([messages]))
    except Exception:
        cb = AsyncIteratorCallbackHandler()
        llm = make_llm(fallback_model)
        task = asyncio.create_task(llm.agenerate([messages]))

    buffer = ""
    async for token in cb.aiter():
        buffer += token
        if buffer.endswith(("\n", "。", ".", "…", "!", "?", ")", "]")):
            line = buffer
            buffer = ""
            if output_format.lower() == "md":
                line = _format_md_stream(line)
            if not _THOUGHT_PAT.match(line.strip()):
                yield f"data: {line}\n\n"

    if buffer.strip():
        line = buffer
        if output_format.lower() == "md":
            line = _format_md_stream(line)
        if not _THOUGHT_PAT.match(line.strip()):
            yield f"data: {line}\n\n"

    try:
        await task
    except Exception as e:
        if "not found" in str(e).lower() or "model \"" in str(e).lower():
            cb2 = AsyncIteratorCallbackHandler()
            llm2 = make_llm(fallback_model)
            task2 = asyncio.create_task(llm2.agenerate([messages]))
            buffer2 = ""
            async for token in cb2.aiter():
                buffer2 += token
                if buffer2.endswith(("\n", "。", ".", "…", "!", "?", ")", "]")):
                    line = buffer2
                    buffer2 = ""
                    if output_format.lower() == "md":
                        line = _format_md_stream(line)
                    if not _THOUGHT_PAT.match(line.strip()):
                        yield f"data: {line}\n\n"
            if buffer2.strip():
                line = buffer2
                if output_format.lower() == "md":
                    line = _format_md_stream(line)
                if not _THOUGHT_PAT.match(line.strip()):
                    yield f"data: {line}\n\n"
            await task2
        else:
            raise


# =========================================================
# 메인 엔트리 (롱독 Map→Reduce + 슬라이드 목표 힌트)
# =========================================================
async def stream_summary_with_langchain(
    text: str,
    domain: str | None = None,
    longdoc: bool = True,
    output_format: str = "md",
    augment_web: bool = False,
    length: str = "medium",
    tone: str = "neutral",
):
    dom = (domain or _detect_domain(text)).lower()
    if dom not in {"meeting", "code", "paper", "general", "lecture"}:
        dom = "general"

    extra_context = ""
    if augment_web and _is_augmentation_allowed():
        try:
            lang = _detect_lang(text)
            entities = _extract_entities_for_web(text, lang=lang, max_items=int(os.getenv("AUGMENT_MAX_ENTITIES", "5")))
            extra_context = _fetch_wikipedia_summaries(entities, lang=lang, max_sources=int(os.getenv("AUGMENT_MAX_SOURCES", "5")))
        except Exception:
            extra_context = ""

    enable_long = longdoc and len(text or "") > DEFAULT_LONGDOC_CHAR_LIMIT

    if length == 'short':
        token_budget = HF_MAX_NEW_TOKENS_SHORT
    elif length == 'medium':
        token_budget = HF_MAX_NEW_TOKENS_MEDIUM
    else:
        token_budget = HF_MAX_NEW_TOKENS_LONG

    backend = DEFAULT_SUMMARY_BACKEND

    if not enable_long:
        target_slides = max(4, SLIDES_MIN // 2)
        sys_txt = _system_prompt(dom, phase="final", output_format=output_format, length=length)
        user_payload = _compose_user_payload(text, extra_context, output_format, length=length, tone=tone, target_slides=target_slides)
        old_budget = os.environ.get('HF_MAX_NEW_TOKENS')
        os.environ['HF_MAX_NEW_TOKENS'] = str(token_budget)
        try:
            if backend == "ollama":
                async for s in _stream_with_ollama(user_payload, system_text=sys_txt, output_format=output_format):
                    yield s
            else:
                async for s in _stream_with_hf(user_payload, system_text=sys_txt, output_format=output_format):
                    yield s
        finally:
            if old_budget is None:
                os.environ.pop('HF_MAX_NEW_TOKENS', None)
            else:
                os.environ['HF_MAX_NEW_TOKENS'] = old_budget
        return

    # Long-doc: Map→Reduce
    chunks = _chunk_text(
        text,
        chunk_chars=DEFAULT_CHUNK_CHARS,
        overlap=DEFAULT_CHUNK_OVERLAP,
    )
    num_chunks = len(chunks)
    target_slides = max(SLIDES_MIN, min(SLIDES_MAX, num_chunks))

    # Map
    map_sys = _system_prompt(dom, phase="map", output_format=output_format, length=length)
    partials: list[str] = []
    for idx, ch in enumerate(chunks, 1):
        try:
            map_input = _compose_user_payload(
                f"[Chunk {idx}/{num_chunks}]\n{ch}",
                "",
                output_format,
                length=length,
                tone=tone,
                target_slides=min(3, max(1, SLIDES_MIN // max(2, num_chunks)))
            )
            part = await _hf_generate_once(map_sys, map_input, max_new_tokens=HF_MAP_MAX_NEW_TOKENS)
        except Exception:
            part = ch[:800]
        partials.append(f"[Chunk {idx}]\n{part.strip()}")

    reduce_text = "\n\n".join(partials)
    reduce_sys = _system_prompt(dom, phase="reduce", output_format=output_format, length=length)
    reduce_input = _compose_user_payload(reduce_text, extra_context, output_format, length=length, tone=tone, target_slides=target_slides)

    old_budget = os.environ.get('HF_MAX_NEW_TOKENS')
    os.environ['HF_MAX_NEW_TOKENS'] = str(token_budget)
    try:
        if backend == "ollama":
            async for s in _stream_with_ollama(reduce_input, system_text=reduce_sys, output_format=output_format):
                yield s
        else:
            async for s in _stream_with_hf(reduce_input, system_text=reduce_sys, output_format=output_format):
                yield s
    finally:
        if old_budget is None:
            os.environ.pop('HF_MAX_NEW_TOKENS', None)
        else:
            os.environ['HF_MAX_NEW_TOKENS'] = old_budget


# =========================================================
# H1 제거 유틸 (저장 전 위생 처리)
# =========================================================
def _strip_top_level_h1_outside_code(s: str) -> str:
    if not s:
        return s
    parts = re.split(r'(```[\s\S]*?```)', s)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'(?m)^[ \t]*#\s+.*\n?', '', parts[i])
    return ''.join(parts)


# =========================================================
# (엔드포인트에서 재사용) 슬라이드/정규화 헬퍼
# =========================================================
def count_slides(md: str) -> int:
    if not md:
        return 0
    return len(re.findall(r'(?mi)^###\s*슬라이드\s*\d+', md))

def normalize_and_renumber_slides(md: str) -> str:
    """'# 슬라이드1' 같은 난형식도 '### 슬라이드 n: 제목'으로 통일하고 번호 재배열."""
    if not md:
        return md
    lines = md.splitlines()
    out = []
    has_section = any(re.match(r'(?mi)^##\s*슬라이드\s*$', ln.strip()) for ln in lines)
    inserted_section = False

    slide_idx = 0
    header_pat = re.compile(r'(?mi)^\s*#{1,3}\s*슬라이드\s*(\d+)?\s*[:：]?\s*(.*)$')

    for ln in lines:
        m = header_pat.match(ln.strip())
        if m:
            if not has_section and not inserted_section:
                out.append("## 슬라이드")
                out.append("")
                inserted_section = True
            slide_idx += 1
            title = (m.group(2) or "").strip()
            if not title:
                title = "요약"
            out.append(f"### 슬라이드 {slide_idx}: {title}")
        else:
            out.append(ln)

    if (not has_section) and (slide_idx == 0):
        return md
    return "\n".join(out)
