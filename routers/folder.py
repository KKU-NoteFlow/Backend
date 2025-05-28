from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from db import get_db
from models.folder import Folder
from models.note import Note
from schemas.folder import FolderCreate, FolderResponse, FolderUpdate
from schemas.note import NoteResponse
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1", tags=["Folders"])

@router.get("/folders", response_model=List[FolderResponse])
def list_folders(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    # 1. 유저 폴더·노트 전부 불러오기
    all_folders = db.query(Folder).filter(Folder.user_id == user.u_id).all()
    all_notes   = db.query(Note).filter(Note.user_id == user.u_id).all()

    # 2. 노트들을 folder_id 기준으로 그룹핑
    folder_note_map: dict[int, list[Note]] = {}
    for n in all_notes:
        folder_note_map.setdefault(n.folder_id, []).append(n)

    # 3. 각 Folder 객체에 children, notes 속성 추가
    id_map = {f.id: f for f in all_folders}
    for f in all_folders:
        setattr(f, 'children', [])
        setattr(f, 'notes', folder_note_map.get(f.id, []))

    # 4. 트리 구조 생성
    roots: list[Folder] = []
    for f in all_folders:
        if f.parent_id and f.parent_id in id_map:
            id_map[f.parent_id].children.append(f)
        else:
            roots.append(f)

    return roots

@router.post("/folders", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
def create_folder(
    req: FolderCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    new = Folder(user_id=user.u_id, name=req.name, parent_id=req.parent_id)
    db.add(new)
    db.commit()
    db.refresh(new)
    # 빈 리스트 속성 초기화
    setattr(new, 'children', [])
    setattr(new, 'notes', [])
    return new

@router.patch("/folders/{folder_id}", response_model=FolderResponse)
def update_folder(
    folder_id: int,
    req: FolderUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.user_id == user.u_id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    if req.name is not None:
        folder.name = req.name
    if req.parent_id is not None:
        folder.parent_id = req.parent_id
    db.commit()
    db.refresh(folder)
    # 업데이트 후에도 children, notes 채워주기
    setattr(folder, 'children', [])
    setattr(folder, 'notes', [])
    return folder

@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.user_id == user.u_id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    db.delete(folder)
    db.commit()
    return
