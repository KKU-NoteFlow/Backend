# Backend/routers/checklist.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db import get_db
from models.checklist import Checklist
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1/checklists", tags=["Checklists"])

@router.get("")
def list_checklists(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """현재 사용자 체크리스트 전체 목록 반환(최신순)."""
    items = (
        db.query(Checklist)
        .filter(Checklist.user_id == user.u_id)
        .order_by(Checklist.id.desc())
        .all()
    )
    return [
        {"id": it.id, "title": it.title, "is_clear": int(bool(it.is_clear))}
        for it in items
    ]

@router.post("", status_code=status.HTTP_201_CREATED)
def create_checklist(title: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    obj = Checklist(user_id=user.u_id, title=title, is_clear=False)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "title": obj.title, "is_clear": int(bool(obj.is_clear))}

@router.patch("/{checklist_id}/clear")
def set_clear_state(checklist_id: int, is_clear: bool, db: Session = Depends(get_db), user=Depends(get_current_user)):
    obj = db.query(Checklist).filter(Checklist.id == checklist_id, Checklist.user_id == user.u_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Checklist not found")
    obj.is_clear = bool(is_clear)
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "is_clear": int(bool(obj.is_clear))}
