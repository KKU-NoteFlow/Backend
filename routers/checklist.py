# 변경/설명:
# - POST /checklists : 생성 전용
# - PATCH /checklists/{id}/clear : is_clear 0/1 설정
# - get_current_user는 user.u_id 제공 가정
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.checklist import ChecklistCreate, ChecklistClearUpdate
from app.models import Checklist  # Checklist(u_id, checklist_title, is_clear)
from app.dependencies import get_db, get_current_user

router = APIRouter(prefix="/checklists", tags=["checklists"])

@router.post("", status_code=status.HTTP_201_CREATED)
def create_checklist(
    req: ChecklistCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    obj = Checklist(
        u_id=user.u_id,                 # ← 프로젝트의 사용자 키에 맞게
        checklist_title=req.checklist_title,
        is_clear=0                      # 기본 0(미완)
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "checklist_title": obj.checklist_title, "is_clear": obj.is_clear}

@router.patch("/{checklist_id}/clear")
def set_clear_state(
    checklist_id: int,
    req: ChecklistClearUpdate,          # {"is_clear": 0 | 1}
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    obj = (
        db.query(Checklist)
          .filter(Checklist.id == checklist_id, Checklist.u_id == user.u_id)
          .first()
    )
    if not obj:
        raise HTTPException(status_code=404, detail="Checklist not found")
    obj.is_clear = int(req.is_clear)    # 0/1 저장
    db.commit()
    db.refresh(obj)
    return {"id": obj.id, "is_clear": obj.is_clear}