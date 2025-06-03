# ~/noteflow/Backend/utils/llm.py

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

_MODEL_NAME = "Qwen/Qwen3-8B"

# 전역 변수: 최초에는 토크나이저/모델이 None
_tokenizer = None
_model = None

def _load_model():
    """
    summarize_with_qwen3()가 최초 호출될 때만 Qwen3-8B 모델과 토크나이저를 메모리에 로드합니다.
    """
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        # 1) Config 불러와서 parallel_style 지정
        config = AutoConfig.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=True
        )
        # 반드시 "auto"로 지정 (NoneType 오류 방지)
        config.parallel_style = "auto"

        # 2) 토크나이저 로드
        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=True
        )

        # 3) 모델 로드 시 config 인자 추가
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            config=config,            # custom config 전달
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        _model.eval()


def summarize_with_qwen3(
    text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.6
) -> str:
    """
    - 한국어 문서를 간결하고 핵심적으로 요약
    - 반환값: 요약된 한국어 문자열
    """
    # 모델/토크나이저가 아직 로드되지 않았다면, 이 시점에만 로드
    if _model is None or _tokenizer is None:
        _load_model()

    # Chat-format prompt 생성
    messages = [
        {
            "role": "system",
            "content": (
                "당신은 한국어 문서를 간결하고 핵심적으로 요약하는 전문가입니다. "
                "요약 외에는 절대 다른 말을 하지 마세요."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    # tokenizer.apply_chat_template()를 통해 모델 친화적인 프롬프트 생성
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 입력 토크나이즈 후 모델 디바이스로 이동
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    # 모델 generate 호출
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        do_sample=False,              # 안정적인 요약을 위해 샘플링 끄기
        eos_token_id=_tokenizer.eos_token_id
    )

    # 입력 프롬프트 뒤에 생성된 토큰만 디코딩
    gen_tokens = outputs[0].tolist()[len(inputs.input_ids[0]):]
    decoded = _tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return decoded.strip()
