from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# =========================================================
# 문제 생성(Question Generation, QG) 스키마
# =========================================================

class QuestionItem(BaseModel):
    """생성된 문제 하나에 대한 데이터 모델"""
    question: str = Field(..., description="질문 내용")
    answer: str = Field(..., description="정답 내용")
    # 객관식일 경우 4개의 보기, 주관식일 경우 빈 리스트
    options: List[str] = Field(default_factory=list, description="질문 보기를 포함하는 리스트") 

class QuestionGenerationRequest(BaseModel):
    """문제 생성 요청 시 사용되는 입력 모델"""
    text: str = Field(..., description="문제를 생성할 기반 텍스트 (LLM 요약 결과 등)")
    num_questions: int = Field(5, description="생성할 문제 개수 (1-10)", ge=1, le=10)
    question_type: Literal["multiple_choice", "short_answer"] = Field(
        "multiple_choice", description="생성할 문제 유형"
    )
    language: Literal["ko", "en"] = Field("ko", description="텍스트의 언어 (ko 또는 en)")

class QuestionGenerationResponse(BaseModel):
    """문제 생성 결과 응답 모델"""
    questions: List[QuestionItem] = Field(..., description="생성된 문제 리스트")