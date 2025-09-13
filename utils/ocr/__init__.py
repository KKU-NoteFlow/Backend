"""
utils.ocr 패키지

추가/변경 요약
- 공통 OCR 파이프라인 진입점(run_pipeline)을 외부에 노출
- 이미지/PDF/DOC/DOCX/HWP를 단일 인터페이스로 처리
"""

from .ocr_core import run_pipeline, detect_type

__all__ = [
    "run_pipeline",
    "detect_type",
]

