from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import re, asyncio

_THOUGHT_PAT = re.compile(
    r"^\s*(okay|let\s*me|i\s*need\s*to|first[, ]|then[, ]|next[, ]|in summary|먼저|그\s*다음|요약하면)",
    re.I,
)

async def stream_summary_with_langchain(text: str):
    """
    LangChain + Ollama에서 토큰을 비동기로 받아
    SSE("data: ...\\n\\n") 형식으로 yield 하는 async generator
    """
    # 1) LangChain용 콜백 핸들러
    cb = AsyncIteratorCallbackHandler()

    # 2) Ollama Chat 모델 (streaming=True)
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model="qwen3:8b",
        streaming=True,
        callbacks=[cb],
        temperature=0.6,
    )

    # 3) 프롬프트
    messages = [
        SystemMessage(
            content="다음 텍스트를 한국어로 간결하게 요약하세요. "
                    "사고 과정(Chain‑of‑Thought)은 절대 출력하지 마세요./no_think"
        ),
        HumanMessage(content=text),
    ]

    # 4) LLM 호출 비동기 실행
    task = asyncio.create_task(llm.agenerate([messages]))

    buffer = ""
    async for token in cb.aiter():
        buffer += token
        if buffer.endswith(("\n", "。", ".", "…")):
            line = buffer.strip()
            buffer = ""

            if not _THOUGHT_PAT.match(line):
                yield f"data: {line}\n\n"          # SSE 청크 전송

    await task  # 예외 전파
