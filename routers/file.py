# routers/file.py
import os
import io
import whisper
model = whisper.load_model("base")
from datetime import datetime
import numpy as np
from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image

from db import get_db
from models.file import File as FileModel
from models.note import Note as NoteModel
from utils.jwt_utils import get_current_user

# 추가: 파일명 인코딩용
import urllib.parse

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

    user_dir = os.path.join(BASE_UPLOAD_DIR, str(current_user.u_id))
    os.makedirs(user_dir, exist_ok=True)

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

    try:
        with open(saved_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

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

    # 원본 파일명 UTF-8 URL 인코딩 처리
    quoted_name = urllib.parse.quote(file_obj.original_name, safe='')
    content_disposition = f"inline; filename*=UTF-8''{quoted_name}"  

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
    • ocr_file: 이미지 파일(UploadFile)
    • 1) EasyOCR로 기본 텍스트 추출 (GPU 모드)
    • 2) TrOCR 4개 모델로 OCR 수행 (모두 GPU)
    • 3) 가장 긴 결과를 최종 OCR 결과로 선택
    • 4) Note로 저장 및 결과 반환
    """

    # 1) 이미지 로드 (PIL)
    contents = await ocr_file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {e}")

    # 2) EasyOCR로 텍스트 추출
    try:
        image_np = np.array(image)
        easy_results = reader.readtext(image_np)  # GPU 모드 사용
        easy_text = " ".join([res[1] for res in easy_results])
    except Exception:
        easy_text = ""

    # 3) TrOCR 모델 4개로 OCR 수행 (모두 GPU input)
    hf_texts: List[str] = []
    try:
        out1 = hf_trocr_printed(image)
        if isinstance(out1, list) and "generated_text" in out1[0]:
            hf_texts.append(out1[0]["generated_text"].strip())

        out2 = hf_trocr_handwritten(image)
        if isinstance(out2, list) and "generated_text" in out2[0]:
            hf_texts.append(out2[0]["generated_text"].strip())

        out3 = hf_trocr_small_printed(image)
        if isinstance(out3, list) and "generated_text" in out3[0]:
            hf_texts.append(out3[0]["generated_text"].strip())

        out4 = hf_trocr_large_printed(image)
        if isinstance(out4, list) and "generated_text" in out4[0]:
            hf_texts.append(out4[0]["generated_text"].strip())
    except Exception:
        # TrOCR 중 오류 발생 시 무시하고 계속 진행
        pass

    # 4) 여러 OCR 결과 병합: 가장 긴 문자열을 최종 ocr_text로 선택
    candidates = [t for t in [easy_text] + hf_texts if t and t.strip()]
    if not candidates:
        raise HTTPException(status_code=500, detail="텍스트를 인식할 수 없습니다.")

    ocr_text = max(candidates, key=lambda s: len(s))

    # 5) 새 노트 생성 및 DB에 저장
    try:
        new_note = NoteModel(
            user_id=current_user.u_id,
            folder_id=folder_id,
            title="OCR 결과",
            content=ocr_text  # **원본 OCR 텍스트만 저장**
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


@router.post("/audio")
async def upload_audio_and_transcribe(
    file: UploadFile = File(...),
    note_id: Optional[int] = Form(None),
    folder_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    # 📁 저장 경로 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user{user.u_id}_{timestamp}_{file.filename}"
    save_dir = os.path.join(BASE_UPLOAD_DIR, str(user.u_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 📥 파일 저장
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # ✅ note_id가 있으면 folder_id는 무시
    folder_id_to_use = folder_id if note_id is None else None

    # 📦 files 테이블에 기록
    new_file = FileModel(
        user_id=user.u_id,
        folder_id=folder_id_to_use,
        original_name=filename,
        saved_path=save_path,
        content_type="audio"
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    # 🧠 STT 처리
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(save_path, language="ko")
        transcript = result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 처리 실패: {e}")

    # 📝 노트 처리
    if note_id:
        # 기존 노트에 텍스트 추가
        note = db.query(NoteModel).filter(
            NoteModel.id == note_id,
            NoteModel.user_id == user.u_id
        ).first()

        if not note:
            raise HTTPException(status_code=404, detail="해당 노트를 찾을 수 없습니다.")

        note.content = (note.content or "") + "\n\n" + transcript
        note.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(note)

    else:
        # 새 노트 생성
        new_note = NoteModel(
            user_id=user.u_id,
            folder_id=folder_id_to_use,
            title="녹음 텍스트",
            content=transcript
        )
        db.add(new_note)
        db.commit()
        db.refresh(new_note)

    return {
        "message": "STT 및 노트 저장 완료",
        "transcript": transcript
    }