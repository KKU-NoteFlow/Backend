
from pydantic import BaseModel, conint

class ChecklistCreate(BaseModel):
    checklist_title: str

class ChecklistClearUpdate(BaseModel):
    is_clear: conint(ge=0, le=1)  # 0 또는 1만 허용