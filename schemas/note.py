from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class NoteCreate(BaseModel):
    title: str
    content: Optional[str] = None
    folder_id: Optional[int] = None

class FavoriteUpdate(BaseModel):
    is_favorite: bool

class NoteResponse(BaseModel):
    id: int
    user_id: int
    folder_id: Optional[int]
    title: str
    content: Optional[str]
    is_favorite: bool
    last_accessed: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
