# ~/noteflow/Backend/routers/file.py

import os
import io
import numpy as np
from typing import Optional, List
from urllib.parse import quote

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image

from db import get_db
from models.file import File as FileModel
from models.note import Note as NoteModel
from utils.jwt_utils import get_current_user

# -------------------------------
# 1) EasyOCR 라이브러리 임포트 (GPU 모드 활성화)
# -------------------------------
import easyocr
reader = easyocr.Reader(["ko", "en"], gpu=True)

# -------------------------------
# 2) Hugging Face TrOCR 모델용 파이프라인 (GPU 사용)
# -------------------------------
from transformers import pipeline

hf_trocr_printed = pipeline(
    "image-to-text",
    model="microsoft/trocr-base-printed",
    device=0,
    trust_remote_code=True
)
hf_trocr_handwritten = pipeline(
    "image-to-text",
    model="microsoft/trocr-base-handwritten",
    device=0,
    trust_remote_code=True
)
hf_trocr_small_printed = pipeline(
    "image-to-text",
    model="microsoft/trocr-small-printed",
    device=0,
    trust_remote_code=True
)
hf_trocr_large_printed = pipeline(
    "image-to-text",
    model="microsoft/trocr-large-printed",
    device=0,
    trust_remote_code=True
)

# 업로드 디렉토리 설정
BASE_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "uploads"
)
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/api/v1/files", tags=["Files"])


@router.post(
    "/upload",
    summary="폴더에 파일 업로드",
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
    folder_id: Optional[int] = Form(None),
    upload_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    orig_filename: str = upload_file.filename or "unnamed"
    content_type: str = upload_file.content_type or "application/octet-stream"

    # 사용자별 디렉토리 생성
    user_dir = os.path.join(BASE_UPLOAD_DIR, str(current_user.u_id))
    os.makedirs(user_dir, exist_ok=True)

    # 원본 파일명 그대로 저장 (동명이인 방지)
    saved_filename = orig_filename
    saved_path = os.path.join(user_dir, saved_filename)
    if os.path.exists(saved_path):
        name, ext = os.path.splitext(orig_filename)
        counter = 1
        while True:
            candidate = f"{name}_{counter}{ext}"
            candidate_path = os.path.join(user_dir, candidate)
            if not os.path.exists(candidate_path):
                saved_filename = candidate
                saved_path = candidate_path
                break
            counter += 1

    # 파일 저장
    try:
        with open(saved_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    # DB에 메타데이터 기록
    new_file = FileModel(
        user_id=current_user.u_id,
        folder_id=folder_id,
        original_name=orig_filename,
        saved_path=saved_path,
        content_type=content_type
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    base_url = os.getenv("BASE_API_URL", "http://localhost:8000")
    download_url = f"{base_url}/api/v1/files/download/{new_file.id}"

    return {
        "file_id": new_file.id,
        "url": download_url,
        "original_name": new_file.original_name,
        "folder_id": new_file.folder_id,
        "content_type": new_file.content_type,
        "created_at": new_file.created_at
    }


@router.get(
    "/list/{folder_id}",
    response_model=List[dict],
    summary="특정 폴더에 속한 파일 목록 조회"
)
def list_files_in_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    files = (
        db.query(FileModel)
        .filter(
            FileModel.folder_id == folder_id,
            FileModel.user_id == current_user.u_id
        )
        .order_by(FileModel.created_at.desc())
        .all()
    )
    return [
        {
            "file_id": f.id,
            "original_name": f.original_name,
            "content_type": f.content_type,
            "created_at": f.created_at
        }
        for f in files
    ]


@router.get(
    "/download/{file_id}",
    summary="파일 미리보기 (인증 없이 바로 열림)"
)
def download_file(
    file_id: int,
    db: Session = Depends(get_db),
):
    file_obj = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not file_obj:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    file_path = file_obj.saved_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="서버에 파일이 존재하지 않습니다.")

    # original_name 을 percent-encoding 해서 ASCII 만으로 헤더 구성
    filename_quoted = quote(file_obj.original_name)
    content_disposition = f"inline; filename*=UTF-8''{filename_quoted}"

    return FileResponse(
        path=file_path,
        media_type=file_obj.content_type,
        headers={"Content-Disposition": content_disposition}
    )


@router.post(
    "/ocr",
    summary="이미지 OCR → 텍스트 변환 후 노트 생성",
    response_model=dict
)
async def ocr_and_create_note(
    ocr_file: UploadFile = File(...),
    folder_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    • EasyOCR + TrOCR 모델로 이미지에서 텍스트 추출
    • 가장 긴 결과를 선택해 새 노트로 저장
    """
    # 1) 이미지 로드
    contents = await ocr_file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {e}")

    # 2) EasyOCR
    try:
        image_np = np.array(image)
        easy_results = reader.readtext(image_np)
        easy_text = " ".join([res[1] for res in easy_results])
    except Exception:
        easy_text = ""

    # 3) TrOCR 4개 모델
    hf_texts: List[str] = []
    try:
        for pipe in (
            hf_trocr_printed,
            hf_trocr_handwritten,
            hf_trocr_small_printed,
            hf_trocr_large_printed
        ):
            out = pipe(image)
            if isinstance(out, list) and "generated_text" in out[0]:
                hf_texts.append(out[0]["generated_text"].strip())
    except Exception:
        pass

    # 4) 가장 긴 결과 선택
    candidates = [t for t in [easy_text] + hf_texts if t and t.strip()]
    if not candidates:
        raise HTTPException(status_code=500, detail="텍스트를 인식할 수 없습니다.")
    ocr_text = max(candidates, key=len)

    # 5) Note 생성
    try:
        new_note = NoteModel(
            user_id=current_user.u_id,
            folder_id=folder_id,
            title="OCR 결과",
            content=ocr_text
        )
        db.add(new_note)
        db.commit()
        db.refresh(new_note)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"노트 저장 실패: {e}")

    return {
        "note_id": new_note.id,
        "text": ocr_text
    }
