from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, text
from sqlalchemy.orm import relationship
from .base import Base

class Folder(Base):
    __tablename__ = "folder"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey("user.u_id", ondelete="CASCADE"), nullable=False)
    name       = Column(String(100), nullable=False)
    parent_id  = Column(Integer, ForeignKey("folder.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"),
                        onupdate=text("CURRENT_TIMESTAMP"))

    # relations
    user     = relationship("User")
    parent   = relationship("Folder", remote_side=[id], backref="children")
    notes    = relationship("Note", back_populates="folder", cascade="all, delete")
