"""
utils/ocr/ocr_core.py

추가/변경
- 파일 타입 판별(확장자 우선, MIME 보조) 및 통합 OCR 파이프라인(run_pipeline) 구현
- 이미지: pytesseract 기본 OCR, (기존) EasyOCR/TrOCR는 가능 시 보조로 시도하여 최적 텍스트 선택
- PDF: pdf2image(convert_from_path, dpi=200)로 페이지 이미지를 생성하여 페이지별 OCR
- DOC/DOCX: LibreOffice(soffice --headless)로 PDF로 변환 후 PDF 파이프라인 재사용
- HWP: hwp5txt로 텍스트 추출(성공 시 page=1로 results에 추가), 실패 시 warnings 기록
- 대용량 제어: MAX_PAGES(기본 50)까지 처리하고 잘린 경우 warnings 기록
- 예외는 raise하지 않고 results=[], warnings로 사유를 담아 상위가 200으로 응답할 수 있게 함
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image

from .converters import (
    save_bytes_to_temp,
    pdf_to_images,
    office_to_pdf,
    hwp_to_text,
)


# 지원 확장자 세트
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}
DOC_EXTS = {".doc", ".docx"}
HWP_EXTS = {".hwp"}


def detect_type(filename: str, content_type: Optional[str]) -> str:
    """확장자 기반 타입 판별, MIME은 보조.
    반환: "image" | "pdf" | "docx" | "hwp" | "unknown"
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in DOC_EXTS:
        return "docx"  # 내부적으로 DOC/DOCX를 동일 경로로 처리
    if ext in HWP_EXTS:
        return "hwp"

    # MIME 보조 판단(간단히)
    if content_type:
        ct = content_type.lower()
        if ct.startswith("image/"):
            return "image"
        if ct == "application/pdf":
            return "pdf"
        if ct in ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            return "docx"
        if "hwp" in ct:
            return "hwp"
    return "unknown"


def _ocr_image_pytesseract(img: Image.Image, langs: str, warnings: List[str]) -> str:
    """pytesseract를 사용하여 이미지에서 텍스트 추출.
    주: 시스템에 tesseract OCR 엔진 및 언어 데이터가 설치되어 있어야 함.
    """
    try:
        import pytesseract
        text = pytesseract.image_to_string(img, lang=langs)
        return text.strip()
    except Exception as e:
        warnings.append(f"pytesseract OCR 실패: {e}")
        return ""


def _ocr_image_legacy(img: Image.Image, warnings: List[str]) -> str:
    """기존 이미지 OCR(EasyOCR + TrOCR) 로직 재사용.
    - 환경/의존성에 따라 실패할 수 있으므로 예외는 warnings에만 기록.
    - 기존 구현과 동일하게 가장 긴 텍스트를 선택.
    """
    try:
        import numpy as np
        import easyocr
        from transformers import pipeline
    except Exception as e:
        warnings.append(f"기존 OCR 모듈(EasyOCR/TrOCR) 사용 불가: {e}")
        return ""

    try:
        # EasyOCR
        reader = easyocr.Reader(["ko", "en"], gpu=False)
        image_np = np.array(img.convert("RGB"))
        easy_results = reader.readtext(image_np)
        easy_text = " ".join([res[1] for res in easy_results])
    except Exception as e:
        warnings.append(f"EasyOCR 실패: {e}")
        easy_text = ""

    hf_texts: List[str] = []
    try:
        for model_name in (
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-small-printed",
            "microsoft/trocr-large-printed",
        ):
            try:
                pipe = pipeline("image-to-text", model=model_name, trust_remote_code=True)
                out = pipe(img)
                if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
                    hf_texts.append(out[0]["generated_text"].strip())
            except Exception as e:
                warnings.append(f"TrOCR({model_name}) 실패: {e}")
    except Exception as e:
        warnings.append(f"TrOCR 파이프라인 초기화 실패: {e}")

    candidates = [t for t in [easy_text] + hf_texts if t and t.strip()]
    if not candidates:
        return ""
    return max(candidates, key=len)


def _ocr_image_best(img: Image.Image, langs: str, warnings: List[str]) -> str:
    """
    변경(모델 우선): 기존(EasyOCR/TrOCR) → pytesseract 순으로 시도하고 더 긴 텍스트 선택.
    - 서버에 Tesseract가 없어도 동작하도록 모델 기반 경로를 우선.
    """
    legacy_text = _ocr_image_legacy(img, warnings)
    tesseract_text = _ocr_image_pytesseract(img, langs, warnings)

    candidates = [t for t in [legacy_text, tesseract_text] if t]
    if not candidates:
        return ""
    return max(candidates, key=len)


def run_pipeline(
    filename: str,
    mime: Optional[str],
    data: bytes,
    langs: str = "kor+eng",
    max_pages: int = 50,
) -> Dict:
    """공통 OCR 파이프라인

    반환 JSON 스키마:
    {
      "filename": str,
      "mime": str | null,
      "page_count": int,
      "results": [{"page": int, "text": str}],
      "warnings": [str]
    }

    예외는 raise하지 않고 warnings에만 기록 후 results를 비워서 반환.
    """
    warnings: List[str] = []
    results: List[Dict] = []
    page_count = 0

    ftype = detect_type(filename, mime)

    try:
        if ftype == "image":
            # 단일 이미지 → 페이지 1로 간주
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                warnings.append(f"이미지 열기 실패: {e}")
                img = None

            if img is not None:
                text = _ocr_image_best(img, langs, warnings)
                page_count = 1
                results.append({"page": 1, "text": text or ""})

        elif ftype == "pdf":
            # 변경: PyMuPDF(fitz) 우선 사용 → 네이티브 텍스트, 없으면 렌더링 후 모델 OCR
            images: List[Image.Image] = []
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=data, filetype="pdf")
                total = doc.page_count
                if total > max_pages:
                    warnings.append(f"페이지가 {max_pages}장을 초과하여 앞 {max_pages}페이지만 처리합니다.")
                limit = min(total, max_pages)
                for i in range(limit):
                    page = doc.load_page(i)
                    txt = (page.get_text("text") or "").strip()
                    if txt:
                        results.append({"page": i + 1, "text": txt})
                    else:
                        # 이미지 렌더링 후 모델 OCR
                        try:
                            mat = fitz.Matrix(2, 2)  # ~144 DPI 정도
                            pix = page.get_pixmap(matrix=mat)
                            mode = "RGBA" if pix.alpha else "RGB"
                            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                            if mode == "RGBA":
                                img = img.convert("RGB")
                            images.append(img)
                        except Exception as e:
                            warnings.append(f"PDF 페이지 렌더링 실패(page {i+1}): {e}")
                page_count = limit
            except Exception as e:
                warnings.append(f"PyMuPDF 처리 실패: {e}")
                # 대체 경로: pdf2image(poppler 필요)
                pdf_path = save_bytes_to_temp(data, suffix=".pdf")
                try:
                    images = pdf_to_images(pdf_path, dpi=200)
                except Exception as ee:
                    warnings.append(f"PDF를 이미지로 변환 실패: {ee}")
                    images = []
                finally:
                    try:
                        os.remove(pdf_path)
                    except Exception:
                        pass
                total = len(images)
                if total > max_pages:
                    warnings.append(f"페이지가 {max_pages}장을 초과하여 앞 {max_pages}페이지만 처리합니다.")
                    images = images[:max_pages]
                page_count = len(images)

            # 이미지에 대해 모델 OCR 수행 (필요한 페이지만)
            for idx, img in enumerate(images, start=1):
                text = _ocr_image_best(img, langs, warnings)
                results.append({"page": idx, "text": text or ""})

        elif ftype == "docx":
            # 변경: .docx는 python-docx로 네이티브 텍스트 추출 우선, .doc는 LibreOffice 변환
            ext = os.path.splitext(filename or "")[1].lower()
            if ext == ".docx":
                try:
                    from docx import Document  # python-docx
                    doc = Document(io.BytesIO(data))
                    paras = []
                    for p in doc.paragraphs:
                        if p.text:
                            paras.append(p.text)
                    text = "\n".join(paras).strip()
                    if text:
                        results.append({"page": 1, "text": text})
                        page_count = 1
                    else:
                        warnings.append("DOCX에서 추출된 텍스트가 없습니다.")
                except Exception as e:
                    warnings.append(f"python-docx 처리 실패: {e}")
            else:
                # 구형 .doc → LibreOffice로 PDF 변환 후 OCR
                in_path = save_bytes_to_temp(data, suffix=ext or ".doc")
                outdir = os.path.dirname(in_path)
                pdf_path: Optional[str] = None
                try:
                    pdf_path = office_to_pdf(in_path, outdir)
                    # PDF 처리 동일 (PyMuPDF 경로 우선)
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        total = doc.page_count
                        if total > max_pages:
                            warnings.append(f"페이지가 {max_pages}장을 초과하여 앞 {max_pages}페이지만 처리합니다.")
                        limit = min(total, max_pages)
                        for i in range(limit):
                            page = doc.load_page(i)
                            txt = (page.get_text("text") or "").strip()
                            if txt:
                                results.append({"page": i + 1, "text": txt})
                            else:
                                mat = fitz.Matrix(2, 2)
                                pix = page.get_pixmap(matrix=mat)
                                mode = "RGBA" if pix.alpha else "RGB"
                                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                                if mode == "RGBA":
                                    img = img.convert("RGB")
                                t = _ocr_image_best(img, langs, warnings)
                                results.append({"page": i + 1, "text": t or ""})
                        page_count = limit
                    except Exception as e:
                        warnings.append(f"DOC→PDF 처리 후 읽기 실패: {e}")
                except Exception as e:
                    warnings.append(f"DOC 변환 실패: {e}")
                finally:
                    try:
                        os.remove(in_path)
                    except Exception:
                        pass
                    if pdf_path:
                        try:
                            os.remove(pdf_path)
                        except Exception:
                            pass

        elif ftype == "hwp":
            # HWP → hwp5txt 1차 시도. 성공 시 page=1
            in_path = save_bytes_to_temp(data, suffix=".hwp")
            try:
                text = hwp_to_text(in_path)
                results.append({"page": 1, "text": (text or "").strip()})
                page_count = 1
            except Exception as e:
                warnings.append(f"HWP 텍스트 추출 실패: {e}")
            finally:
                try:
                    os.remove(in_path)
                except Exception:
                    pass

        else:
            warnings.append("지원되지 않는 파일 형식입니다.")

    except Exception as e:
        # 상위에서 200으로 내려줄 수 있도록 전체 예외 흡수
        warnings.append(f"파이프라인 실행 오류: {e}")

    return {
        "filename": filename,
        "mime": mime,
        "page_count": page_count,
        "results": results,
        "warnings": warnings,
    }
