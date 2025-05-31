# Backend/routers/folder.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from db import get_db
from models.folder import Folder
from models.note import Note
from schemas.folder import FolderCreate, FolderResponse, FolderUpdate
from schemas.note import NoteResponse
from utils.jwt_utils import get_current_user

router = APIRouter(prefix="/api/v1", tags=["Folders"])

def get_all_descendant_folder_ids(db: Session, parent_id: int, user_id: int) -> List[int]:
    """
    ● 폴더를 삭제할 때, 하위 폴더(자손)들까지 전부 삭제하기 위해
      재귀적으로 descendant ID를 수집합니다.
    """
    result: List[int] = []
    stack = [parent_id]

    while stack:
        current = stack.pop()
        result.append(current)
        children = (
            db
            .query(Folder.id)
            .filter(Folder.parent_id == current, Folder.user_id == user_id)
            .all()
        )
        stack.extend([child.id for child in children])

    return result


@router.get(
    "/folders",
    response_model=List[FolderResponse],
    summary="유저의 모든 폴더(트리 구조) 및 폴더별 노트 리스트 반환"
)
def list_folders(
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    1) 해당 유저의 모든 폴더와 노트를 가져옵니다.
    2) 노트들은 folder_id 기준으로 그룹핑합니다.
    3) 각 폴더 객체에 children(하위폴더)와 notes(폴더 내 노트) 속성을 동적으로 붙입니다.
    4) 부모가 없는(root) 폴더만 뽑아서 트리 형태(List[Folder])로 반환합니다.
    """
    all_folders = db.query(Folder).filter(Folder.user_id == user.u_id).all()
    all_notes   = db.query(Note).filter(Note.user_id == user.u_id).all()

    # (2) 노트들을 folder_id 기준으로 그룹핑
    folder_note_map: dict[int, List[Note]] = {}
    for n in all_notes:
        folder_note_map.setdefault(n.folder_id, []).append(n)

    # (3) Folder 객체에 children, notes 속성 추가
    id_map = {f.id: f for f in all_folders}
    for f in all_folders:
        setattr(f, 'children', [])               # children: List[Folder]
        setattr(f, 'notes', folder_note_map.get(f.id, []))  # notes: List[Note]

    # (4) 트리 형태 구성
    roots: List[Folder] = []
    for f in all_folders:
        if f.parent_id is not None and f.parent_id in id_map:
            id_map[f.parent_id].children.append(f)
        else:
            roots.append(f)

    return roots


@router.post(
    "/folders",
    response_model=FolderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="새 폴더 생성"
)
def create_folder(
    req: FolderCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    • req.name (문자열, 필수)
    • req.parent_id (정수 or null) - 최상위 폴더면 None
    """
    new_folder = Folder(
        user_id = user.u_id,
        name    = req.name,
        parent_id = req.parent_id
    )
    db.add(new_folder)
    db.commit()
    db.refresh(new_folder)

    # children, notes 속성 초기화
    setattr(new_folder, 'children', [])
    setattr(new_folder, 'notes', [])

    return new_folder


@router.patch(
    "/folders/{folder_id}",
    response_model=FolderResponse,
    summary="폴더 이름 변경 및/또는 부모 폴더 이동"
)
def update_folder(
    folder_id: int,
    req: FolderUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    ● 이름(name) 변경이 있으면 바꿔주고,
    ● 부모폴더(parent_id) 변경이 있으면 바꿔줍니다.
      (자신을 자신의 하위로 지정하는 경우를 방지하려면
       프론트에서 부모ID 검증 혹은 백엔드에서 추가 검증 필요)
    """
    folder = (
        db
        .query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == user.u_id)
        .first()
    )
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    if req.name is not None:
        folder.name = req.name

    if req.parent_id is not None:
        folder.parent_id = req.parent_id

    db.commit()
    db.refresh(folder)

    # 응답 시 children, notes는 빈 배열로 초기화
    setattr(folder, 'children', [])
    setattr(folder, 'notes', [])

    return folder


@router.delete(
    "/folders/{folder_id}",
    summary="폴더 및 모든 하위 폴더·노트 일괄 삭제"
)
def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """
    1) 해당 폴더(및 하위폴더)의 ID 목록을 재귀적으로 수집
    2) 그 ID들에 속한 노트들을 모두 삭제
    3) 그 ID들에 속한 폴더들을 모두 삭제
    """
    folder = (
        db
        .query(Folder)
        .filter(Folder.id == folder_id, Folder.user_id == user.u_id)
        .first()
    )
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    all_folder_ids = get_all_descendant_folder_ids(db, folder_id, user.u_id)

    # (2) 해당 폴더들 안의 모든 노트 삭제
    db.query(Note).filter(Note.folder_id.in_(all_folder_ids)).delete(synchronize_session=False)

    # (3) 해당 폴더들 삭제
    db.query(Folder).filter(Folder.id.in_(all_folder_ids)).delete(synchronize_session=False)

    db.commit()
    return {"message": f"Deleted folder and its {len(all_folder_ids)-1} subfolders."}
