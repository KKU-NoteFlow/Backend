from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import asyncio

# qg_service.py에서 정의할 서비스 함수를 가져옵니다.
from utils.qg_service import generate_questions_from_text

# schemas에서 정의할 모델을 가져옵니다.
from schemas import QuestionGenerationRequest, QuestionGenerationResponse, QuestionItem

router = APIRouter(prefix="/qg", tags=["Question Generation"])

@router.post(
    "/generate",
    response_model=QuestionGenerationResponse,
    summary="요약 텍스트를 기반으로 예상 문제(객관식)를 생성합니다."
)
async def generate_questions_endpoint(request: QuestionGenerationRequest):
    """
    제공된 텍스트(주로 LLM 요약 결과)를 사용하여 예상 문제를 생성합니다.
    """
    if not request.text or len(request.text.strip()) < 100:
        raise HTTPException(
            status_code=400,
            detail="문제 생성을 위한 텍스트 내용이 너무 짧거나 비어 있습니다. (최소 100자 이상 필요)"
        )
    
    try:
        # 비동기적으로 문제 생성 서비스 호출
        questions = await generate_questions_from_text(
            text=request.text,
            num_questions=request.num_questions,
            question_type=request.question_type,
            language=request.language
        )
        
        return QuestionGenerationResponse(
            questions=questions
        )
    except Exception as e:
        # 실제 환경에서는 로깅이 필요합니다.
        print(f"문제 생성 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail="문제 생성 서비스 처리 중 서버 오류가 발생했습니다."
        )