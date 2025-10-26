from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class OCRResultItem(BaseModel):
    page: int
    text: str


class OCRResponse(BaseModel):
    # 신규 필드(추가/변경): 공통 파이프라인 메타
    filename: str
    mime: Optional[str] = None
    page_count: int
    results: List[OCRResultItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # 하위 호환(변경 전 응답 유지): 기존 이미지 OCR 응답
    note_id: Optional[int] = None
    text: Optional[str] = None

