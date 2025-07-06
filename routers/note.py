import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import traceback

from db import get_db, SessionLocal
from models.note import Note
from schemas.note import NoteCreate, NoteUpdate, NoteResponse, FavoriteUpdate
from utils.jwt_utils import get_current_user
from fastapi.responses import StreamingResponse 
from utils.llm import stream_summary_with_langchain

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

router = APIRouter(prefix="/api/v1", tags=["Notes"])


# 1) 모든 노트 조회
@router.get("/notes", response_model=List[NoteResponse])
def list_notes(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    return (
        db.query(Note)
        .filter(Note.user_id == user.u_id)
        .order_by(Note.created_at.desc())
        .all()
    )


# 2) 최근 접근한 노트 조회 (상위 10개)
@router.get("/notes/recent", response_model=List[NoteResponse])
def recent_notes(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    return (
        db.query(Note)
        .filter(Note.user_id == user.u_id, Note.last_accessed.isnot(None))
        .order_by(Note.last_accessed.desc())
        .limit(10)
        .all()
    )


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
    return note


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

    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return note


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
    return note


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
    return note

def save_summary(note_id: int, text: str):
    db2 = SessionLocal()
    try:
        tgt = db2.query(Note).filter(Note.id == note_id).first()
        if tgt:
            tgt.content = text
            tgt.updated_at = datetime.utcnow()
            db2.commit()
    finally:
        db2.close()

@router.post("/notes/{note_id}/summarize")
async def summarize_stream_langchain(
    note_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    async def event_gen():
        parts = []
        async for sse in stream_summary_with_langchain(note.content):
            parts.append(sse.removeprefix("data: ").strip())
            yield sse.encode()                    
        full = "".join(parts).strip()
        if full:
            background_tasks.add_task(save_summary, note.id, full)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )