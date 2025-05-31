# Backend/schemas/folder.py

from pydantic import BaseModel
from typing import Optional, List
from schemas.note import NoteResponse

class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None

class FolderUpdate(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[int] = None

class FolderResponse(BaseModel):
    id: int
    user_id: int
    name: str
    parent_id: Optional[int]
    children: List['FolderResponse'] = []
    notes: List[NoteResponse] = []

    class Config:
        from_attributes = True

FolderResponse.update_forward_refs()
