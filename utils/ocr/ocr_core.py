"""
utils/ocr/ocr_core.py

- 파일 타입 판별(detect_type)
- 통합 OCR 파이프라인(run_pipeline)
- 이미지: Tesseract + EasyOCR + HuggingFace TrOCR 조합 (긴 텍스트 선택)
- PDF: PyMuPDF → 네이티브 텍스트 추출, 없으면 이미지 렌더링 후 OCR
- DOCX: python-docx, DOC: LibreOffice 변환 후 OCR
- HWP: hwp5txt 사용
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Optional
from PIL import Image

from .converters import (
    save_bytes_to_temp,
    pdf_to_images,
    office_to_pdf,
    hwp_to_text,
)

# 지원 확장자
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}
DOC_EXTS = {".doc", ".docx"}
HWP_EXTS = {".hwp"}


# ─────────────────────────────────────────────
# 타입 판별
# ─────────────────────────────────────────────
def detect_type(filename: str, content_type: Optional[str]) -> str:
    """확장자 기반 타입 판별, MIME은 보조."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in DOC_EXTS:
        return "docx"
    if ext in HWP_EXTS:
        return "hwp"

    if content_type:
        ct = content_type.lower()
        if ct.startswith("image/"):
            return "image"
        if ct == "application/pdf":
            return "pdf"
        if ct in (
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            return "docx"
        if "hwp" in ct:
            return "hwp"
    return "unknown"


# ─────────────────────────────────────────────
# OCR Backends
# ─────────────────────────────────────────────
def _ocr_image_pytesseract(img: Image.Image, langs: str, warnings: List[str]) -> str:
    """pytesseract OCR"""
    try:
        import pytesseract
        return pytesseract.image_to_string(img, lang=langs).strip()
    except Exception as e:
        warnings.append(f"pytesseract OCR 실패: {e}")
        return ""


def _ocr_image_legacy(img: Image.Image, warnings: List[str]) -> str:
    """EasyOCR + HuggingFace TrOCR"""
    try:
        import numpy as np
        import easyocr
        from transformers import pipeline
    except Exception as e:
        warnings.append(f"EasyOCR/TrOCR import 실패: {e}")
        return ""

    # EasyOCR
    try:
        reader = easyocr.Reader(["ko", "en"], gpu=False)
        image_np = np.array(img.convert("RGB"))
        easy_results = reader.readtext(image_np)
        easy_text = " ".join([res[1] for res in easy_results])
    except Exception as e:
        warnings.append(f"EasyOCR 실패: {e}")
        easy_text = ""

    # HuggingFace TrOCR
    hf_texts: List[str] = []
    for model_name in [
        "microsoft/trocr-base-printed",
        "microsoft/trocr-base-handwritten",
    ]:
        try:
            pipe = pipeline("image-to-text", model=model_name, trust_remote_code=True)
            out = pipe(img)
            if isinstance(out, list) and out and "generated_text" in out[0]:
                hf_texts.append(out[0]["generated_text"].strip())
        except Exception as e:
            warnings.append(f"TrOCR({model_name}) 실패: {e}")

    candidates = [t for t in [easy_text] + hf_texts if t]
    return max(candidates, key=len) if candidates else ""


def _ocr_image_best(img: Image.Image, langs: str, warnings: List[str]) -> str:
    """모든 OCR 경로 실행 후 가장 긴 텍스트 선택"""
    legacy = _ocr_image_legacy(img, warnings)
    tess = _ocr_image_pytesseract(img, langs, warnings)
    candidates = [t for t in [legacy, tess] if t]
    return max(candidates, key=len) if candidates else ""


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    filename: str,
    mime: Optional[str],
    data: bytes,
    langs: str = "kor+eng",
    max_pages: int = 50,
) -> Dict:
    """
    OCR 실행 파이프라인
    반환 JSON:
    {
      "filename": str,
      "mime": str,
      "page_count": int,
      "results": [{"page": int, "text": str}],
      "warnings": [str]
    }
    """
    warnings: List[str] = []
    results: List[Dict] = []
    page_count = 0

    ftype = detect_type(filename, mime)

    try:
        if ftype == "image":
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                warnings.append(f"이미지 열기 실패: {e}")
                img = None
            if img:
                text = _ocr_image_best(img, langs, warnings)
                results.append({"page": 1, "text": text})
                page_count = 1

        elif ftype == "pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=data, filetype="pdf")
                total = doc.page_count
                limit = min(total, max_pages)
                if total > max_pages:
                    warnings.append(f"앞 {max_pages}페이지만 처리")
                for i in range(limit):
                    page = doc.load_page(i)
                    txt = (page.get_text("text") or "").strip()
                    if txt:
                        results.append({"page": i + 1, "text": txt})
                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        mode = "RGBA" if pix.alpha else "RGB"
                        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                        if mode == "RGBA":
                            img = img.convert("RGB")
                        t = _ocr_image_best(img, langs, warnings)
                        results.append({"page": i + 1, "text": t})
                page_count = limit
            except Exception as e:
                warnings.append(f"PyMuPDF 실패: {e}")
                # fallback → pdf2image
                pdf_path = save_bytes_to_temp(data, suffix=".pdf")
                try:
                    images = pdf_to_images(pdf_path, dpi=200)
                    if len(images) > max_pages:
                        warnings.append(f"앞 {max_pages}페이지만 처리")
                        images = images[:max_pages]
                    for idx, img in enumerate(images, 1):
                        t = _ocr_image_best(img, langs, warnings)
                        results.append({"page": idx, "text": t})
                    page_count = len(images)
                finally:
                    try:
                        os.remove(pdf_path)
                    except:
                        pass

        elif ftype == "docx":
            ext = os.path.splitext(filename or "")[1].lower()
            if ext == ".docx":
                try:
                    from docx import Document
                    doc = Document(io.BytesIO(data))
                    paras = [p.text for p in doc.paragraphs if p.text]
                    text = "\n".join(paras).strip()
                    results.append({"page": 1, "text": text})
                    page_count = 1
                except Exception as e:
                    warnings.append(f"python-docx 실패: {e}")
            else:
                in_path = save_bytes_to_temp(data, suffix=".doc")
                try:
                    pdf_path = office_to_pdf(in_path, os.path.dirname(in_path))
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    return run_pipeline(os.path.basename(pdf_path), "application/pdf", pdf_bytes, langs, max_pages)
                except Exception as e:
                    warnings.append(f"DOC 변환 실패: {e}")
                finally:
                    try:
                        os.remove(in_path)
                    except:
                        pass

        elif ftype == "hwp":
            in_path = save_bytes_to_temp(data, suffix=".hwp")
            try:
                text = hwp_to_text(in_path)
                results.append({"page": 1, "text": text.strip()})
                page_count = 1
            except Exception as e:
                warnings.append(f"HWP 처리 실패: {e}")
            finally:
                try:
                    os.remove(in_path)
                except:
                    pass

        else:
            warnings.append("지원되지 않는 파일 형식")

    except Exception as e:
        warnings.append(f"파이프라인 실행 오류: {e}")

    return {
        "filename": filename,
        "mime": mime,
        "page_count": page_count,
        "results": results,
        "warnings": warnings,
    }
