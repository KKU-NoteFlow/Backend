from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime 

from db import get_db
from models.note import Note
from schemas.note import NoteCreate, NoteResponse
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1", tags=["Notes"])

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



@router.patch("/notes/{note_id}", response_model=NoteResponse)
def update_note_folder(
    note_id: int,
    req: NoteCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(
      Note.id == note_id, Note.user_id == user.u_id
    ).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    note.folder_id = req.folder_id
    db.commit()
    db.refresh(note)
    return note


@router.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(
      Note.id == note_id, Note.user_id == user.u_id
    ).first()
    if not note:
        raise HTTPException(404, "Note not found")
    # 조회 시 last_accessed 갱신
    note.last_accessed = datetime.utcnow()
    db.commit()
    db.refresh(note)
    return note
