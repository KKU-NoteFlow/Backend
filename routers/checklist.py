from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from db import get_db
from models.checklist import Checklist
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1/checklists", tags=["Checklists"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_checklist(
    title: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    obj = Checklist(u_id=user.u_id, checklist_title=title, is_clear=False)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "title": obj.checklist_title, "is_clear": obj.is_clear}


@router.get("", status_code=status.HTTP_200_OK)
def get_checklists(
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    objs = db.query(Checklist).filter(Checklist.u_id == user.u_id).all()
    return [
        {"id": obj.id, "title": obj.checklist_title, "is_clear": obj.is_clear}
        for obj in objs
    ]


@router.patch("/{checklist_id}/clear", status_code=status.HTTP_200_OK)
def set_clear_state(
    checklist_id: int,
    is_clear: bool,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    obj = db.query(Checklist).filter(
        Checklist.id == checklist_id,
        Checklist.u_id == user.u_id
    ).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Checklist not found")
    obj.is_clear = bool(is_clear)
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "title": obj.checklist_title, "is_clear": obj.is_clear}