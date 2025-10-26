import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import traceback

from db import get_db, SessionLocal
from models.note import Note
from models.file import File as FileModel
from schemas.note import NoteCreate, NoteUpdate, NoteResponse, FavoriteUpdate, NoteFile
from utils.jwt_utils import get_current_user
from utils.llm import stream_summary_with_langchain

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

router = APIRouter(prefix="/api/v1", tags=["Notes"])

# 환경변수에서 BASE_API_URL 가져와 파일 다운로드 URL 구성
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# 공통: Note → NoteResponse 직렬화 + files 채우기
# ─────────────────────────────────────────────
def serialize_note(db: Session, note: Note) -> NoteResponse:
    """
    Note ORM 객체를 NoteResponse로 변환하면서
    note_id로 연결된 File들을 찾아 files 배열에 채워 넣는다.
    """
    # 파일 목록 조회 (노트 첨부)
    files = (
        db.query(FileModel)
        .filter(FileModel.note_id == note.id, FileModel.user_id == note.user_id)
        .order_by(FileModel.created_at.desc())
        .all()
    )

    file_items: List[NoteFile] = []
    for f in files:
        file_items.append(
            NoteFile(
                file_id=f.id,
                original_name=f.original_name,
                content_type=f.content_type,
                url=f"{BASE_API_URL}/api/v1/files/download/{f.id}",
                created_at=f.created_at,
            )
        )

    # Pydantic 모델 생성 (from_attributes=True로 기본 필드 매핑)
    data = NoteResponse.model_validate(note, from_attributes=True)
    # files 필드 교체
    data.files = file_items
    return data


# 1) 모든 노트 조회
@router.get("/notes", response_model=List[NoteResponse])
def list_notes(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    notes = (
        db.query(Note)
        .filter(Note.user_id == user.u_id)
        .order_by(Note.created_at.desc())
        .all()
    )
    # 각 노트의 files도 채워 반환
    return [serialize_note(db, n) for n in notes]


# 2) 최근 접근한 노트 조회 (상위 10개)
@router.get("/notes/recent", response_model=List[NoteResponse])
def recent_notes(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    notes = (
        db.query(Note)
        .filter(Note.user_id == user.u_id, Note.last_accessed.isnot(None))
        .order_by(Note.last_accessed.desc())
        .limit(10)
        .all()
    )
    return [serialize_note(db, n) for n in notes]


# 3) 노트 생성
@router.post("/notes", response_model=NoteResponse)
def create_note(
    req: NoteCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = Note(
        user_id=user.u_id,
        folder_id=req.folder_id,
        title=req.title,
        content=req.content
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return serialize_note(db, note)


# 4) 노트 수정 (제목/내용/폴더)
@router.patch("/notes/{note_id}", response_model=NoteResponse)
def update_note(
    note_id: int,
    req: NoteUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if req.title is not None:
        note.title = req.title
    if req.content is not None:
        note.content = req.content
    if req.folder_id is not None:
        note.folder_id = req.folder_id
    if req.is_favorite is not None:
        note.is_favorite = req.is_favorite

    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return serialize_note(db, note)


# 5) 노트 단일 조회 (마지막 접근 시간 업데이트 포함)
@router.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.last_accessed = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return serialize_note(db, note)


# 6) 노트 삭제
@router.delete("/notes/{note_id}")
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return {"message": "Note deleted successfully"}


# 7) 즐겨찾기 토글
@router.patch("/notes/{note_id}/favorite", response_model=NoteResponse)
def toggle_favorite(
    note_id: int,
    req: FavoriteUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.is_favorite = req.is_favorite
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return serialize_note(db, note)


# ─────────────────────────────────────────────
# (참고) 요약 스트리밍 API - 완료 후에도 serialize_note 사용 안 함
#        (요약은 새 노트를 생성하고 SSE로 알림만 보냄)
# ─────────────────────────────────────────────
@router.post("/notes/{note_id}/summarize")
async def summarize_stream_langchain(
    note_id: int,
    background_tasks: BackgroundTasks,
    domain: str | None = Query(default=None, description="meeting | code | paper | general | auto(None)"),
    longdoc: bool = Query(default=True, description="Enable long-document map→reduce"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    async def event_gen():
        parts = []
        async for sse in stream_summary_with_langchain(note.content, domain=domain, longdoc=longdoc):
            parts.append(sse.removeprefix("data: ").strip())
            yield sse.encode()
        full = "".join(parts).strip()
        if full:
            # Create a new summary note in the same folder with title '<original>요약'
            title = (note.title or "").strip() + "요약"
            if len(title) > 255:
                title = title[:255]
            new_note = Note(
                user_id=user.u_id,
                folder_id=note.folder_id,
                title=title,
                content=full,
            )
            db.add(new_note)
            db.commit()
            db.refresh(new_note)
            try:
                # Optional: notify created note id
                yield f"data: SUMMARY_NOTE_ID:{new_note.id}\n\n".encode()
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
