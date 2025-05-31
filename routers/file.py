# Backend/routers/file.py

import os
import urllib.parse
import io
from typing import List, Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db import get_db
from models.file import File as FileModel
from utils.jwt_utils import get_current_user
from models.user import User

# PIL(Image) & Transformers(TrOCR + Summarization) import
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline                            # Summarization을 위해 pipeline 가져옴
)

# -------------------------------
# 1) TrOCR(OCR) 모델 로드 (글로벌)
# -------------------------------
# ↓↓↓ 기존 microsoft/trocr-base-printed 대신 한국어·영어 모두 잘 인식하는 ko-trocr 모델 사용
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
# ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
ocr_model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
# ↑↑↑ 변경 끝

# -------------------------------
# 2) Summarization 모델 로드 (글로벌)
#    - 여기서는 BART 계열의 사전학습 요약 모델을 예시로 사용
#    - “facebook/bart-large-cnn”을 불러와 pipeline 구성
# -------------------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    device=-1  # CPU만 사용할 경우 -1. GPU 사용 시 0 또는 적절한 번호로 변경
)
# ※ GPU를 쓰고 싶으면 device=0(or 1,2,…) 로 설정하면 GPU 0번 디바이스 사용

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
    current_user: User = Depends(get_current_user)
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

    return {
        "file_id": new_file.id,
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
    current_user: User = Depends(get_current_user)
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

    filename_star = urllib.parse.quote(file_obj.original_name, safe='')
    content_disposition = f"inline; filename*=UTF-8''{filename_star}"

    return FileResponse(
        path=file_path,
        media_type=file_obj.content_type,
        headers={"Content-Disposition": content_disposition}
    )


@router.post(
    "/ocr",
    summary="이미지 OCR → 텍스트 변환 및 요약",
    response_model=dict
)
async def ocr_and_summarize(
    ocr_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    • ocr_file: 이미지 파일(UploadFile)
    • 1) TrOCR 모델로 텍스트 인식 (한글·영어 모두 인식하는 ko-trocr 사용)
    • 2) 인식된 텍스트를 Summarization 모델로 전달하여 요약
    • 결과: {"text": "원본 OCR 텍스트", "summary": "요약된 텍스트"} 형태로 반환
    """
    # ----------
    # 1) 이미지 처리 → OCR 텍스트 추출
    # ----------
    contents = await ocr_file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {e}")

    # 2) ko-trocr 프로세서로 전처리 (Tensor 형태로 변환)
    inputs = processor(images=image, return_tensors="pt")

    # 3) OCR 모델 추론 (한글·영어 모두 인식)
    generated_ids = ocr_model.generate(**inputs)

    # 4) 디코딩하여 원본 텍스트 추출
    ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ----------
    # 2) Summarization 모델로 요약 수행
    # ----------
    try:
        # Summarization pipeline 호출
        # max_length/min_length는 필요에 따라 조정
        summary_list = summarizer(
            ocr_text,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        # pipeline 결과는 list of dict: [{"summary_text": "..."}]
        summarized_text = summary_list[0]["summary_text"]
    except Exception as e:
        summarized_text = ""
        print(f"[OCR & Summarization] 요약 중 오류 발생: {e}")

    return {
        "text": ocr_text,
        "summary": summarized_text
    }
