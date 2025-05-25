from pydantic import BaseModel
from typing import Optional, List

class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None

class FolderResponse(BaseModel):
    id: int
    user_id: int
    name: str
    parent_id: Optional[int]
    children: List['FolderResponse'] = []

    class Config:
        from_attributes = True

FolderResponse.update_forward_refs()
