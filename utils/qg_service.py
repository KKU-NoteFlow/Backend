# utils/qg_service.py
from __future__ import annotations
import os, json, re, asyncio
from typing import List, Literal, Optional, Dict, Any

# HF Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# schemas에서 정의할 모델을 가져옵니다.
from schemas import QuestionItem 

# =========================================================
# 설정값 (Qwen3-4B-Instruct-2507 모델 사용)
# =========================================================
QG_MODEL_NAME = os.getenv("QG_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507") 
QG_MAX_NEW_TOKENS = int(os.getenv("QG_MAX_NEW_TOKENS", "4096"))
QG_TEMPERATURE = float(os.getenv("QG_TEMPERATURE", "0.2"))
HF_API_TOKEN = os.getenv("HF_API_TOKEN") or None

_QG_MODEL = None
_QG_TOKENIZER = None

# =========================================================
# 모델 로딩 유틸 (멀티 GPU 분산 및 CPU fallback)
# =========================================================
def _resolve_dtype():
    """bf16 지원 시 bf16, 아니면 fp16 사용"""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def _available_gpus():
    """사용 가능한 GPU 인덱스 리스트 반환"""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _load_qg_model():
    """Qwen3-4B 모델을 GPU 여러 개 및 CPU에 분산 로드합니다."""
    global _QG_MODEL, _QG_TOKENIZER
    if _QG_MODEL is not None and _QG_TOKENIZER is not None:
        return _QG_MODEL, _QG_TOKENIZER

    torch_dtype = _resolve_dtype()
    load_in_4bit = os.getenv("HF_LOAD_IN_4BIT", "false").lower() in ("1", "true", "yes")

    print(f"QG 모델 로딩 중: {QG_MODEL_NAME}, 4bit={load_in_4bit}")

    hub_kwargs = {"trust_remote_code": True}
    if HF_API_TOKEN:
        hub_kwargs["token"] = HF_API_TOKEN

    try:
        # 1. 토크나이저 로드
        _QG_TOKENIZER = AutoTokenizer.from_pretrained(QG_MODEL_NAME, **hub_kwargs)

        # 2. 모델 로드 설정
        kwargs = dict(hub_kwargs)
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        # 3. GPU 분산 메모리 설정
        if torch.cuda.is_available():
            gpu_list = _available_gpus()
            print(f"사용 가능한 GPU 목록: {gpu_list}")

            # GPU별 메모리 용량 비율로 분배 (0번은 적게, 2,3번은 많이)
            max_memory = {}
            for i in gpu_list:
                if i == 0:
                    max_memory[i] = "4GiB"   # 이미 다른 작업 중일 가능성
                else:
                    max_memory[i] = "20GiB"  # 여유 GPU는 풀로 사용
            max_memory["cpu"] = "30GiB"

            kwargs["max_memory"] = max_memory
            device_map = "auto"

            print(f"GPU 및 CPU 분산 배치 적용: {max_memory}")
        else:
            print("⚠️ CUDA 비활성화됨 — CPU에서 로드합니다.")
            device_map = {"": "cpu"}

        # 4. 모델 로드 (분산 배치 적용)
        _QG_MODEL = AutoModelForCausalLM.from_pretrained(
            QG_MODEL_NAME,
            device_map=device_map,
            **kwargs
        )

        print("✅ QG 모델 로드 완료.")
        return _QG_MODEL, _QG_TOKENIZER

    except torch.cuda.OutOfMemoryError as oom_e:
        print(f"⚠️ GPU 메모리 부족 — CPU로 fallback합니다: {oom_e}")
        _QG_MODEL = AutoModelForCausalLM.from_pretrained(
            QG_MODEL_NAME,
            device_map={"": "cpu"},
            **hub_kwargs
        )
        return _QG_MODEL, _QG_TOKENIZER

    except Exception as e:
        print(f"QG 모델 로딩 실패: {e}")
        raise RuntimeError(f"QG 모델 로딩 실패: {QG_MODEL_NAME} (오류 내용: {e})") from e


# =========================================================
# 텍스트 전처리 / 프롬프트 빌드
# =========================================================
def _extract_key_points(full_text: str) -> str:
    """텍스트에서 '핵심 요점' 섹션만 추출합니다."""
    match = re.search(
        r"##\s*(핵심 요점|KEY POINTS|KEY TAKEAWAYS)\s*\n(.*?)(\n##\s*|\Z)", 
        full_text, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return "핵심 요점:\n" + match.group(2).strip()
    print("⚠️ '핵심 요점' 섹션 추출 실패 — 전체 텍스트 기반으로 문제 생성.")
    return full_text


def _build_qg_prompt(tokenizer, system_text: str, user_text: str) -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _get_system_prompt(num: int, q_type: str, lang: str) -> str:
    lang_name = "한국어" if lang == "ko" else "English"
    q_type_desc = "객관식 (4지선다, 정답 포함)" if q_type == "multiple_choice" else "단답형 주관식"

    return (
        f"당신은 '{lang_name}'로 작성된 텍스트를 분석하여 전문가 수준의 교육용 문제를 생성하는 봇입니다. "
        "입력 텍스트의 '핵심 요점' 섹션에 명시된 사실만을 근거로 문제를 생성해야 합니다. "
        f"{num}개의 '{q_type_desc}' 문제를 생성하세요. "
        "출력은 **반드시** JSON 배열 형식만 사용해야 합니다. 다른 설명이나 사족을 추가하지 마세요. "
        
        "\n\nJSON 형식: "
        "[\n  {\n    \"question\": \"[질문 내용]\",\n    \"answer\": \"[정답]\",\n    \"options\": [\"[보기1]\", \"[보기2]\", \"[보기3]\", \"[보기4]\"]  // 객관식일 경우\n  }\n]"
        "\n\n규칙:\n"
        "1. 정답은 반드시 'options' 리스트 내에 포함되어야 합니다.\n"
        "2. 질문은 명확하고, 정답은 '핵심 요점'에 기반해야 합니다.\n"
        "3. 마크다운(```` )이나 다른 꾸밈 없이 순수 JSON 배열만 출력합니다."
    )


# =========================================================
# 메인 문제 생성 함수
# =========================================================
async def generate_questions_from_text(
    text: str,
    num_questions: int,
    question_type: Literal["multiple_choice", "short_answer"],
    language: Literal["ko", "en"],
) -> List[QuestionItem]:
    
    # 1. 모델 로드
    try:
        model, tokenizer = _load_qg_model()
    except RuntimeError:
        raise

    # 2. 텍스트 전처리
    key_points_only = _extract_key_points(text)

    # 3. 프롬프트 구성
    system_prompt = _get_system_prompt(num_questions, question_type, language)
    user_payload = f"다음 '핵심 요점' 텍스트를 기반으로 문제를 생성합니다:\n\n---\n{key_points_only}"
    prompt = _build_qg_prompt(tokenizer, system_prompt, user_payload)

    # 4. LLM 추론
    def _generate_sync():
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            try:
                target_device = next(model.parameters()).device
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            except Exception:
                pass

        gen_kwargs = dict(
            max_new_tokens=QG_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=QG_TEMPERATURE,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    loop = asyncio.get_event_loop()
    raw_json_output = await loop.run_in_executor(None, _generate_sync)

    # 5. JSON 파싱 및 저장
    try:
        if raw_json_output.startswith("```"):
            raw_json_output = re.sub(r"^```json\s*", "", raw_json_output, flags=re.IGNORECASE)
            raw_json_output = re.sub(r"```\s*$", "", raw_json_output)

        json_data: List[Dict[str, Any]] = json.loads(raw_json_output)
        question_items = [QuestionItem(**item) for item in json_data]

        # 파일 저장
        questions_for_file = [{"id": i+1, "question": q.question, "options": q.options} for i, q in enumerate(question_items)]
        answers_for_file = [{"id": i+1, "question_preview": q.question[:50] + "...", "correct_answer": q.answer} for i, q in enumerate(question_items)]

        with open("qg_questions.json", "w", encoding="utf-8") as f:
            json.dump({"questions": questions_for_file}, f, ensure_ascii=False, indent=4)
        with open("qg_answers.json", "w", encoding="utf-8") as f:
            json.dump({"answers": answers_for_file}, f, ensure_ascii=False, indent=4)

        print("[INFO] 문제/답안 파일 저장 완료.")
        return question_items

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}\n원본 출력:\n{raw_json_output}")
        raise ValueError(f"LLM 출력 JSON 파싱 실패: {raw_json_output[:100]}...")

    except Exception as e:
        print(f"최종 데이터 처리 오류: {e}")
        raise
