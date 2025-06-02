import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen3-8B"

# 1) 모델과 토크나이저 한 번만 로드
_tokenizer = AutoTokenizer.from_pretrained(
    _MODEL_NAME,
    trust_remote_code=True
)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
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

    # apply_chat_template에 enable_thinking=False 로 전달
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        do_sample=False,             # 더 안정적인 요약을 위해 샘플링 끄기
        eos_token_id=_tokenizer.eos_token_id
    )

    # 응답 부분만 디코딩
    gen_tokens = outputs[0].tolist()[len(inputs.input_ids[0]):]
    decoded = _tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return decoded.strip()