from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from models.folder import Folder
from schemas.folder import FolderCreate, FolderResponse
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1", tags=["Folders"])

@router.get("/folders", response_model=List[FolderResponse])
def list_folders(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    all_f = db.query(Folder).filter(Folder.user_id == user.u_id).all()
    id_map = {f.id: f for f in all_f}
    for f in all_f:
        setattr(f, 'children', [])
    roots = []
    for f in all_f:
        if f.parent_id and f.parent_id in id_map:
            id_map[f.parent_id].children.append(f)
        else:
            roots.append(f)
    return roots

@router.post("/folders", response_model=FolderResponse)
def create_folder(
    req: FolderCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    new = Folder(user_id=user.u_id, name=req.name, parent_id=req.parent_id)
    db.add(new)
    db.commit()
    db.refresh(new)
    setattr(new, 'children', [])
    return new
