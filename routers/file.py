import os
import re
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query, Response, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db import get_db
from models.file import File as FileModel
from models.note import Note as NoteModel
from utils.jwt_utils import get_current_user
import urllib.parse

# 공통 OCR 파이프라인 (내부 유틸로 위임) — 안전 임포트
try:
    from utils.ocr import run_pipeline, detect_type
except ModuleNotFoundError:
    import sys as _sys, os as _os
    _ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)
    from utils.ocr import run_pipeline, detect_type
from schemas.file import OCRResponse

# 허용 확장자
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALLOWED_PDF_EXTS   = {".pdf"}
ALLOWED_DOC_EXTS   = {".doc", ".docx"}
ALLOWED_HWP_EXTS   = {".hwp"}
ALLOWED_ALL_EXTS   = (ALLOWED_IMAGE_EXTS | ALLOWED_PDF_EXTS | ALLOWED_DOC_EXTS | ALLOWED_HWP_EXTS)

# 업로드 디렉토리
BASE_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/api/v1/files", tags=["Files"])


@router.get("/ocr/diag", summary="OCR 런타임 의존성 진단")
def ocr_dependency_diag():
    import shutil, subprocess
    def which(cmd: str):
        return shutil.which(cmd) is not None
    def run(cmd: list[str]):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
            return out.decode(errors="ignore").strip()
        except Exception as e:
            return f"ERR: {e}"

    tesseract_ok = which("tesseract")
    poppler_ok = which("pdftoppm") or which("pdftocairo")
    soffice_ok = which("soffice") or which("libreoffice")
    hwp5txt_ok = which("hwp5txt")

    langs = None
    tess_ver = None
    if tesseract_ok:
        tess_ver = run(["tesseract", "--version"]).splitlines()[0]
        langs_out = run(["tesseract", "--list-langs"])
        langs = [l.strip() for l in langs_out.splitlines() if l and not l.lower().startswith("list of available")]

    return {
        "tesseract": tesseract_ok,
        "tesseract_version": tess_ver,
        "tesseract_langs": langs,
        "poppler": poppler_ok,
        "libreoffice": soffice_ok,
        "hwp5txt": hwp5txt_ok,
    }


@router.post(
    "/upload",
    summary="폴더/노트에 파일 업로드 (note_id 있으면 노트 본문에도 삽입)",
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
    folder_id: Optional[int] = Form(None),
    note_id: Optional[int] = Form(None),
    upload_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    request: Request = None,
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

    # 저장
    try:
        with open(saved_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")

    # note_id가 있으면 해당 노트 확인
    note_obj = None
    if note_id is not None:
        note_obj = (
            db.query(NoteModel)
            .filter(NoteModel.id == note_id, NoteModel.user_id == current_user.u_id)
            .first()
        )
        if not note_obj:
            raise HTTPException(status_code=404, detail="해당 노트를 찾을 수 없습니다.")

    # DB 메타 기록
    new_file = FileModel(
        user_id=current_user.u_id,
        folder_id=None if note_id else folder_id,
        note_id=note_id,
        original_name=orig_filename,
        saved_path=saved_path,
        content_type=content_type
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    # Prefer request base_url if available; fallback to env
    base_url = (str(request.base_url).rstrip('/') if request else None) or os.getenv("BASE_API_URL", "http://localhost:8000")
    download_url = f"{base_url}/api/v1/files/download/{new_file.id}"
    public_url = None
    if content_type.startswith("image/"):
        public_url = f"{base_url}/static/{current_user.u_id}/{urllib.parse.quote(saved_filename)}"

    # note_id가 있으면 노트 본문에 첨부 링크 삽입
    if note_obj:
        if content_type.startswith("image/"):
            embed = f"\n\n![{new_file.original_name}]({download_url})\n\n"
        elif content_type == "application/pdf":
            embed = f"\n\n[{new_file.original_name}]({download_url}) (PDF 보기)\n\n"
        else:
            embed = f"\n\n[{new_file.original_name}]({download_url})\n\n"
        note_obj.content = (note_obj.content or "") + embed
        db.commit()
        db.refresh(note_obj)

    return {
        "file_id": new_file.id,
        "url": download_url,
        "public_url": public_url,
        "original_name": new_file.original_name,
        "folder_id": new_file.folder_id,
        "note_id": new_file.note_id,
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

    return FileResponse(
        path=file_path,
        media_type=file_obj.content_type,
        filename=file_obj.original_name,
        background=None
    )


# ---------------------------
# 언어코드 유연 처리 유틸
# ---------------------------
LANG_ALIAS = {
    "ko": "kor", "kr": "kor", "korean": "kor",
    "en": "eng", "english": "eng"
}
def normalize_langs(raw: str) -> str:
    if not raw:
        return "kor+eng"
    s = raw.strip().lower().replace(" ", "")
    s = re.sub(r"[,_;]+", "+", s)
    if "+" not in s:
        s = s.replace("koreng", "kor+eng").replace("koen", "ko+en")
    parts = [p for p in s.split("+") if p]
    norm: list[str] = []
    for p in parts:
        norm.append(LANG_ALIAS.get(p, p))
    seen, out = set(), []
    for p in norm:
        if p not in seen:
            seen.add(p); out.append(p)
    return "+".join(out) if out else "kor+eng"


@router.post(
    "/ocr",
    summary="이미지/PDF/DOC/DOCX/HWP OCR → 텍스트 변환 후 노트 생성",
    response_model=OCRResponse
)
async def ocr_and_create_note(
    file: Optional[UploadFile] = File(None, description="기본 업로드 필드명"),
    ocr_file: Optional[UploadFile] = File(None, description="과거 호환 업로드 필드명"),
    folder_id: Optional[int] = Form(None),
    langs: str = Query("kor+eng", description="언어코드"),
    max_pages: int = Query(50, ge=1, le=500, description="최대 페이지 수"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    upload = file or ocr_file
    if upload is None:
        raise HTTPException(status_code=400, detail="파일이 필요합니다.")

    filename = upload.filename or "uploaded"
    mime = upload.content_type or "application/octet-stream"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext and ext not in ALLOWED_ALL_EXTS:
        return OCRResponse(
            filename=filename,
            mime=mime,
            page_count=0,
            results=[],
            warnings=[f"허용되지 않는 확장자({ext})."],
            note_id=None,
            text=None,
        )

    data = await upload.read()
    langs = normalize_langs(langs)

    pipe = run_pipeline(
        filename=filename,
        mime=mime,
        data=data,
        langs=langs,
        max_pages=max_pages,
    )

    merged_text = "\n\n".join([
        item.get("text", "") for item in (pipe.get("results") or []) if item.get("text")
    ]).strip()

    note_id: Optional[int] = None
    if merged_text:
        try:
            base_title = os.path.splitext(filename)[0].strip() or "OCR 결과"
            new_note = NoteModel(
                user_id=current_user.u_id,
                folder_id=folder_id,
                title=base_title,
                content=merged_text,
            )
            db.add(new_note)
            db.commit()
            db.refresh(new_note)
            note_id = new_note.id
        except Exception as e:
            (pipe.setdefault("warnings", [])).append(f"노트 저장 실패: {e}")

    pipe["note_id"] = note_id
    pipe["text"] = merged_text or None
    return pipe


@router.post("/audio")
async def upload_audio_and_transcribe(
    file: UploadFile = File(...),
    note_id: Optional[int] = Form(None),
    folder_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"user{user.u_id}_{timestamp}_{file.filename}"
    save_dir = os.path.join(BASE_UPLOAD_DIR, str(user.u_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    folder_id_to_use = folder_id if note_id is None else None

    new_file = FileModel(
        user_id=user.u_id,
        folder_id=folder_id_to_use,
        note_id=note_id,
        original_name=filename,
        saved_path=save_path,
        content_type="audio"
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(save_path, language="ko")
        transcript = result.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 처리 실패: {e}")

    if note_id:
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
        new_note = NoteModel(
            user_id=user.u_id,
            folder_id=folder_id_to_use,
            title="녹음 텍스트",
            content=transcript
        )
        db.add(new_note)
        db.commit()
        db.refresh(new_note)

    return {"message": "STT 및 노트 저장 완료", "transcript": transcript}


@router.options("/ocr")
def ocr_cors_preflight() -> Response:
    return Response(status_code=200)
