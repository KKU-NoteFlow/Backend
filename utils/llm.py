# [CHANGED] 마크다운 보정 + Markdown 섹션 포맷 + 선택적 웹 보강(위키) + 동일 언어 요약을 포함한 전체 코드
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import re, asyncio, os, threading, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# [CHANGED] 웹 보강용 (옵션)
import requests
from urllib.parse import quote as _urlquote

# =============== 필터: 사고과정 유사 문장 ===============
_THOUGHT_PAT = re.compile(
    r"^\s*(okay|let\s*me|i\s*need\s*to|first[, ]|then[, ]|next[, ]|in summary|먼저|그\s*다음|요약하면)",
    re.I,
)

# =============== 마크다운 보정 유틸 ===============
# [CHANGED] 스트리밍 중 붙어버린 헤더/불릿을 자동 교정
_MD_CODE_FENCE = re.compile(r"(```.*?```|`[^`]*`)", re.S)

def _format_md_stream(s: str) -> str:
    """
    스트리밍 중 합쳐진 마크다운을 사람 읽기 좋게 보정.
    - 헤더(#~######) 앞뒤 빈 줄 보장
    - 헤더 해시 뒤 공백 보장
    - 불릿(- )/번호 목록은 줄 시작으로 강제
    - 코드펜스/인라인코드는 건드리지 않음
    """
    if not s:
        return s

    parts = []
    last = 0
    for m in _MD_CODE_FENCE.finditer(s):
        chunk = s[last:m.start()]
        parts.append(_format_md_plain(chunk))
        parts.append(m.group(0))  # 코드펜스 원형 유지
        last = m.end()
    parts.append(_format_md_plain(s[last:]))

    out = "".join(parts)
    out = re.sub(r"\n{3,}", "\n\n", out)  # 과도한 빈 줄 축소
    return out

def _format_md_plain(s: str) -> str:
    # 헤더 해시 뒤 공백 보장: "##개요" -> "## 개요"
    s = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", s, flags=re.M)

    # 문장 중간 헤더 분리: "...개요## 핵심" -> "...개요\n\n## 핵심"
    s = re.sub(r"(?<!\n)\s*(#{1,6}\s)", r"\n\n\1", s)

    # 헤더 앞뒤 빈 줄 1줄 보장
    s = re.sub(r"(?m)(^#{1,6}\s.*$)", r"\n\1\n", s)

    # 불릿(- )/번호목록이 문장 뒤에 붙으면 줄바꿈
    s = re.sub(r"(?<!\n)\s*(-\s+)", r"\n\1", s)
    s = re.sub(r"(?<!\n)\s*(\d+\.\s+)", r"\n\1", s)

    # 리스트 항목 사이 최소 1줄 유지(너무 붙는 경우)
    s = re.sub(r"(\n[-\d].*\S)(?=[^\n]|$)", r"\1", s)
    return s


# =============== 메인 엔트리 ===============
async def stream_summary_with_langchain(
    text: str,
    domain: str | None = None,
    longdoc: bool = True,
    # [CHANGED] 출력/보강 옵션
    output_format: str = "md",          # "md" | "html"
    augment_web: bool = False,          # True면 위키 요약 보강
):
    """
    한국어/영어 자동 인지 후 '동일 언어'로 요약을 스트리밍합니다.
    - domain: None=자동탐지 | meeting | code | paper | general
    - longdoc: True이면 길이 임계 초과 시 청크(Map)→합본(Reduce) 요약 적용
    - output_format: "md"(권장) 또는 "html"
    - augment_web: True면 Wikipedia REST로 '추가자료' 보강
    """
    dom = (domain or _detect_domain(text)).lower()
    if dom not in {"meeting", "code", "paper", "general"}:
        dom = "general"

    # [CHANGED] 선택적 웹 보강
    extra_context = ""
    if augment_web and _is_augmentation_allowed():
        try:
            lang = _detect_lang(text)  # "ko" | "en"
            entities = _extract_entities_for_web(text, lang=lang, max_items=int(os.getenv("AUGMENT_MAX_ENTITIES", "5")))
            extra_context = _fetch_wikipedia_summaries(entities, lang=lang, max_sources=int(os.getenv("AUGMENT_MAX_SOURCES", "5")))
        except Exception:
            extra_context = ""  # 실패 시 조용히 무시

    # 길이 기준: 대략 3500자 초과 시 청크 요약
    enable_long = longdoc and len(text or "") > int(os.getenv("SUMMARY_LONGDOC_CHAR_LIMIT", "3500"))

    backend = os.getenv("SUMMARY_BACKEND", "hf").lower()
    if not enable_long:
        sys_txt = _system_prompt(dom, phase="final", output_format=output_format)
        user_payload = _compose_user_payload(text, extra_context, output_format)  # [CHANGED]
        if backend == "ollama":
            async for s in _stream_with_ollama(user_payload, system_text=sys_txt, output_format=output_format):
                yield s
        else:
            async for s in _stream_with_hf(user_payload, system_text=sys_txt, output_format=output_format):
                yield s
        return

    # Long-doc: Map (chunk summaries) → Reduce (final synthesis streamed)
    chunks = _chunk_text(
        text,
        chunk_chars=int(os.getenv("SUMMARY_CHUNK_CHARS", "2000")),
        overlap=int(os.getenv("SUMMARY_CHUNK_OVERLAP", "200")),
    )
    map_sys = _system_prompt(dom, phase="map", output_format=output_format)
    partials: list[str] = []
    for idx, ch in enumerate(chunks, 1):
        try:
            map_input = _compose_user_payload(ch, "", output_format)  # [CHANGED]
            part = await _hf_generate_once(map_sys, map_input, max_new_tokens=int(os.getenv("HF_MAP_MAX_NEW_TOKENS", "220")))
        except Exception:
            part = ch[:500]
        partials.append(f"[Chunk {idx}]\n{part.strip()}")

    reduce_text = "\n\n".join(partials)
    reduce_sys = _system_prompt(dom, phase="reduce", output_format=output_format)
    reduce_input = _compose_user_payload(reduce_text, extra_context, output_format)  # [CHANGED]

    if backend == "ollama":
        async for s in _stream_with_ollama(reduce_input, system_text=reduce_sys, output_format=output_format):
            yield s
    else:
        async for s in _stream_with_hf(reduce_input, system_text=reduce_sys, output_format=output_format):
            yield s


# =============== 도메인/언어 감지 ===============
def _detect_domain(t: str) -> str:
    s = (t or "").lower()
    # code-like signals
    if re.search(r"\b(def |class |import |#include|public\s+class|function\s|=>|:=)", s) or re.search(r"```|\bdiff --git\b|\bcommit\b", s):
        return "code"
    # paper-like signals
    if re.search(r"\babstract\b|\bintroduction\b|\bmethod(s)?\b|\bresult(s)?\b|\bconclusion(s)?\b|doi:|arxiv:\d", s):
        return "paper"
    # meeting-like signals (KO/EN keywords)
    if re.search(r"회의|안건|결정|논의|액션 아이템|참석자|회의록|meeting|agenda|minutes|action items|attendees", s):
        return "meeting"
    return "general"

def _detect_lang(t: str) -> str:
    """아주 단순한 언어 감지(영문자/한글자 수 비교). ko/en만 구분."""
    s = t or ""
    en = len(re.findall(r"[A-Za-z]", s))
    ko = len(re.findall(r"[가-힣]", s))
    return "en" if en > ko else "ko"


# =============== 시스템 프롬프트 ===============
# [CHANGED] 출력 포맷(MD/HTML) 지원 + 마크다운 간격 규칙 + 도메인별 포함 요소 힌트
def _system_prompt(domain: str, phase: str = "final", output_format: str = "md") -> str:
    # phase: map | reduce | final
    fmt = output_format.lower()
    base_rules = (
        "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 텍스트의 언어를 감지하고, 반드시 동일한 언어로 작성한다. "
        "핵심 정보(주요 주장/결론, 인물·기관·수치·날짜·지표·범위, 원인↔결과·조건·한계)를 빠짐없이 담되 군더더기와 반복을 제거한다. "
        "추정·가치판단·조언은 덧붙이지 않는다. 사고과정(Chain-of-Thought), 단계 나열, 메타 코멘트는 출력하지 않는다."
    )
    if domain == "meeting":
        include_hint = "결정 사항, 책임자/기한이 명시된 액션, 남은 이슈, 다음 단계"
    elif domain == "code":
        include_hint = "변경 목적/범위, 주요 API/모듈 영향, 호환성/리스크, 마이그레이션 포인트"
    elif domain == "paper":
        include_hint = "문제 정의·동기, 방법 요지, 데이터/세팅, 핵심 수치, 한계/가정"
    else:
        include_hint = "5W1H, 핵심 주장/결과, 중요한 수치·날짜·고유명사, 원인↔결과·조건, 한계/주의점"

    if fmt == "md":
        format_rule = (
            "출력 형식: Markdown. 다음 섹션을 사용하라 — "
            "# 제목, ## 개요, ## 핵심 요점(불릿 3–7개), ## 상세 설명(문단 분리), "
            "## 용어 정리(필요시), ## 한계/주의, ## 할 일(있다면), ## 참고/추가자료(있다면). "
            "각 섹션 제목만 출력하고, 불필요한 프리앰블은 쓰지 마라."
        )
    else:
        format_rule = (
            "출력 형식: HTML fragment. <h1>, <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>만 사용. "
            "<h1>제목</h1>, <h2>개요</h2>, <h2>핵심 요점</h2><ul>…</ul>, <h2>상세 설명</h2>, "
            "<h2>용어 정리</h2>, <h2>한계/주의</h2>, <h2>할 일</h2>, <h2>참고/추가자료</h2>의 순서."
        )

    length_rule = "분량: 원문 대비 약 15–30%. 각 문단은 2–5문장."
    # [CHANGED] 마크다운 간격 규칙 추가
    md_spacing_rule = (
        "마크다운 간격 규칙: 모든 헤더(#, ##, ### 등) 뒤에는 한 칸 공백을 두고, 헤더의 앞뒤에는 빈 줄 1줄을 둔다. "
        "불릿(- )은 항목마다 줄바꿈하고, 서브항목은 들여쓰기 2–4칸을 사용한다."
    )
    web_rule = (
        "추가자료가 주어지면 원문과 교차 검증하여 모순이 있으면 원문을 우선하라. "
        "추가자료의 출처명(예: 위키백과)을 괄호로 간단히 표기할 수 있다."
    )

    if phase == "map":
        scope = "이 청크만 대상으로 섹션 골격을 간략히 채워라. 과도한 요약 금지."
    elif phase == "reduce":
        scope = "아래 청크 요약들을 중복 없이 통합해 일관된 섹션 구성을 완성하라. 흐름(원인→과정→결과)을 유지."
    else:
        scope = "전체 텍스트를 위 섹션 구조에 맞춰 응집력 있게 작성하라. 첫 줄부터 본문만 출력."

    return f"{base_rules}\n포함 우선: {include_hint}\n{format_rule}\n{length_rule}\n{md_spacing_rule}\n{web_rule}\n{scope}"


# =============== HF backend (Transformers) ===============
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
    except Exception:
        model, tok = try_load(fallback)
        _HF_MODEL, _HF_TOKENIZER, _HF_NAME = model, tok, fallback
        return _HF_MODEL, _HF_TOKENIZER


def _build_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # [CHANGED] 백업 프롬프트도 MD/HTML 구조 반영
        return (
            "You are a precise summarizer. Detect the input language (Korean/English) and write the summary in the SAME language. "
            "Preserve key facts (claims, entities, numbers, dates); remove fluff; avoid speculation and chain-of-thought. "
            "Use Markdown sections: # Title, ## Overview, ## Key Points, ## Details, ## Terms, ## Limitations, ## Action Items, ## References.\n\n"
            "Text:\n" + user_text + "\n\nSummary:"
        )


async def _stream_with_hf(text: str, system_text: str | None = None, output_format: str = "md"):
    model, tokenizer = _load_hf_model()

    sys_msg = system_text or (
        # [CHANGED] 기본 시스템 프롬프트: Markdown 섹션 + 동일 언어
        "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 언어를 감지하고 동일 언어로 작성한다. "
        "Markdown 섹션(# 제목, ## 개요, ## 핵심 요점, ## 상세 설명, ## 용어 정리, ## 한계/주의, ## 할 일, ## 참고/추가자료)을 사용한다. "
        "핵심 주장/결과, 인물·기관·수치·날짜, 원인↔결과·조건·한계를 보존하고 군더더기는 제거한다. "
        "추정·가치판단·조언 금지. 사고과정/단계 나열/메타 코멘트 금지. 각 문단 2–5문장."
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
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "320")),  # [CHANGED] 섹션 증가에 맞춰 상향
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        streamer=streamer,
    )
    if gen_kwargs.get("do_sample"):
        gen_kwargs["temperature"] = float(os.getenv("HF_TEMPERATURE", "0.1"))

    def _gen():
        model.generate(**inputs, **gen_kwargs)

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
        # [CHANGED] 플러시 트리거 확장 + 보정기 적용
        if buffer.endswith(("\n", "。", ".", "…", "!", "?", ")", "]")):
            line = buffer  # strip 하지 않음: 줄바꿈 유지
            buffer = ""
            # [CHANGED] 마크다운 보정
            if output_format.lower() == "md":
                line = _format_md_stream(line)
            if not _THOUGHT_PAT.match(line.strip()):
                yield f"data: {line}\n\n"

    # [CHANGED] 잔여 버퍼 마무리 보정
    if buffer.strip():
        line = buffer
        if output_format.lower() == "md":
            line = _format_md_stream(line)
        if not _THOUGHT_PAT.match(line.strip()):
            yield f"data: {line}\n\n"


async def _hf_generate_once(system_text: str, user_text: str, max_new_tokens: int = 256) -> str:
    """Non-streaming single-shot generation (used for chunk map stage)."""
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


# =============== 텍스트 청크 나누기 ===============
def _chunk_text(text: str, chunk_chars: int = 2000, overlap: int = 200) -> list[str]:
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


# =============== Ollama backend ===============
async def _stream_with_ollama(text: str, system_text: str | None = None, output_format: str = "md"):
    # 1) LangChain용 콜백 핸들러
    cb = AsyncIteratorCallbackHandler()

    # 2) Ollama Chat 모델 설정 (환경변수로 조정 가능)
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
        return ChatOllama(
            base_url=base_url,
            model=model_name,
            streaming=True,
            callbacks=[cb],
            temperature=0.4,
            **kwargs,
        )

    llm = make_llm(primary_model)

    # [CHANGED] Ollama 경로의 시스템 프롬프트도 MD 섹션 구조 지시
    messages = [
        SystemMessage(
            content=(
                system_text or (
                    "역할: 너는 사실 보존에 강한 전문 요약가다. 입력 언어를 감지하고 동일 언어로 작성한다. "
                    "Markdown 섹션(# 제목, ## 개요, ## 핵심 요점, ## 상세 설명, ## 용어 정리, ## 한계/주의, ## 할 일, ## 참고/추가자료)을 사용한다. "
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
        # [CHANGED] 플러시 트리거 확장 + 마크다운 보정 적용
        if buffer.endswith(("\n", "。", ".", "…", "!", "?", ")", "]")):
            line = buffer
            buffer = ""
            if output_format.lower() == "md":
                line = _format_md_stream(line)
            if not _THOUGHT_PAT.match(line.strip()):
                yield f"data: {line}\n\n"

    # [CHANGED] 잔여 버퍼 마무리 보정
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
            # [CHANGED] 잔여 버퍼
            if buffer2.strip():
                line = buffer2
                if output_format.lower() == "md":
                    line = _format_md_stream(line)
                if not _THOUGHT_PAT.match(line.strip()):
                    yield f"data: {line}\n\n"
            await task2
        else:
            raise


# =============== 보조 유틸리티 ===============
# [CHANGED] 원문 + (선택) 추가자료를 모델에 전달하기 위한 합본
def _compose_user_payload(main_text: str, extra_context: str, output_format: str) -> str:
    fmt = output_format.lower()
    if fmt == "md":
        if extra_context:
            return (
                "## 원문\n"
                f"{main_text}\n\n"
                "## 추가자료(요약)\n"
                f"{extra_context}\n"
            )
        return main_text
    else:
        if extra_context:
            return f"<h2>원문</h2>\n{main_text}\n\n<h2>추가자료(요약)</h2>\n{extra_context}\n"
        return main_text

def _is_augmentation_allowed() -> bool:
    """환경변수로 보강 ON/OFF 제어. 기본 False."""
    return os.getenv("AUGMENT_WEB", "false").lower() in ("1", "true", "yes")

def _extract_entities_for_web(text: str, lang: str = "ko", max_items: int = 5) -> list[str]:
    """
    매우 가벼운 엔티티 후보 추출:
    - 영문: 대문자로 시작하는 2~4단어 구
    - 한글: 괄호/따옴표 내 주요어 + 2자 이상 단어
    """
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
    """
    간단한 Wikipedia summary 수집.
    - 공개 REST 엔드포인트 사용(무인증)
    - 실패/타임아웃은 건너뜀
    - 결과는 Markdown 불릿로 반환
    """
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
            # ko/en 모두 동일한 표기
            src = "위키백과" if lang == "ko" else "Wikipedia"
            out.append(f"- **{title}** ({src}): {extract}")
            time.sleep(0.05)
        except Exception:
            continue
    return "\n".join(out)
