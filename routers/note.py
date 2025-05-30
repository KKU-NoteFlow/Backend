import os
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime 

from db import get_db
from models.note import Note
from schemas.note import NoteCreate, NoteResponse, FavoriteUpdate
from utils.jwt_utils import get_current_user
from utils.llm import summarize_with_qwen3

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

router = APIRouter(prefix="/api/v1", tags=["Notes"])

# 모든 노트 조회
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

# 최근 접근한 노트 조회
@router.get("/notes/recent", response_model=List[NoteResponse])
def recent_notes(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    return (
        db.query(Note)
        .filter(Note.user_id == user.u_id, Note.last_accessed.isnot(None))
        .order_by(Note.last_accessed.desc())
        .limit(5)
        .all()
    )

# 노트 생성
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

# 노트 수정 (제목/내용/폴더)
@router.patch("/notes/{note_id}", response_model=NoteResponse)
def update_note(
    note_id: int,
    req: NoteCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.title = req.title
    note.content = req.content
    note.folder_id = req.folder_id
    note.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(note)
    return note

# 노트 단일 조회
@router.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(404, "Note not found")

    # 조회 시 마지막 접근 시간 갱신
    note.last_accessed = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return note

# ✅ 노트 삭제
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

@router.post("/notes/{note_id}/summarize", response_model=NoteResponse)
def summarize_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(
        Note.id==note_id, Note.user_id==user.u_id
    ).first()
    if not note:
        raise HTTPException(404, "Note not found")
    original = note.content or ""
    if not original.strip():
        raise HTTPException(400, "내용이 비어 있어 요약할 수 없습니다.")

    # utils/llm.summarize_with_qwen3 호출
    summary = summarize_with_qwen3(original)

    note.content = summary
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return note