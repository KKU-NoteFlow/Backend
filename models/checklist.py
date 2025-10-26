from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, text
from sqlalchemy.orm import relationship
from .base import Base


class Checklist(Base):
    __tablename__ = "checklist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # DB 컬럼명은 u_id 이므로 명시적으로 매핑
    user_id = Column("u_id", Integer, ForeignKey("user.u_id", ondelete="CASCADE"), nullable=False)
    # DB 컬럼명은 checklist_title 이므로 명시적으로 매핑
    title = Column("checklist_title", String(255), nullable=False)
    is_clear = Column(Boolean, nullable=False, server_default=text("0"))

    user = relationship("User", back_populates="checklists")
