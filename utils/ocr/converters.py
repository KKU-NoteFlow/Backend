from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List

from PIL import Image


def save_bytes_to_temp(data: bytes, suffix: str = "") -> str:
    """바이트를 임시 파일로 저장하고 경로를 반환 (호출자가 삭제 책임)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    pdf2image.convert_from_path로 PDF를 PIL 이미지 리스트로 변환.
    (poppler 설치 필요)
    """
    from pdf2image import convert_from_path  # 지연 임포트
    images = convert_from_path(pdf_path, dpi=dpi)
    return images


def office_to_pdf(input_path: str, outdir: str) -> str:
    """
    LibreOffice(soffice)를 사용하여 DOC/DOCX를 PDF로 변환.
    반환: 변환된 PDF 경로
    실패 시 예외 발생(상위에서 warnings 처리)
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("LibreOffice(soffice) 실행 파일을 찾을 수 없습니다.")

    cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", outdir, input_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"LibreOffice 변환 실패: {proc.stderr.decode(errors='ignore')[:300]}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    pdf_path = os.path.join(outdir, f"{base}.pdf")
    if not os.path.exists(pdf_path):
        # 일부 환경에서 출력 파일명이 다르게 생성될 수 있어 재탐색
        candidates = [p for p in os.listdir(outdir) if p.lower().endswith(".pdf")]
        if candidates:
            pdf_path = os.path.join(outdir, candidates[0])
    if not os.path.exists(pdf_path):
        raise RuntimeError("PDF 결과 파일이 생성되지 않았습니다.")
    return pdf_path


def hwp_to_text(input_path: str) -> str:
    """
    hwp5txt(또는 pyhwp)로 HWP 텍스트 추출.
    hwp5txt CLI가 없으면 예외.
    """
    hwp5txt = shutil.which("hwp5txt")
    if not hwp5txt:
        raise RuntimeError("hwp5txt 실행 파일을 찾을 수 없습니다.")
    proc = subprocess.run([hwp5txt, input_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"hwp5txt 추출 실패: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout.decode(errors="ignore")
