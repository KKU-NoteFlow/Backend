from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class Checklist(Base):
    __tablename__ = "checklist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    u_id = Column(Integer, ForeignKey("user.u_id", ondelete="CASCADE"), nullable=False)
    checklist_title = Column(String(255), nullable=False)
    is_clear = Column(Boolean, nullable=False, default=False)

    # 관계 매핑 (User 모델 쪽에 back_populates 필요)
    user = relationship("User", back_populates="checklists")