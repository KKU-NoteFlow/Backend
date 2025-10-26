from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, text
from .base import Base

class File(Base):
    __tablename__ = 'file'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey('user.u_id', ondelete='CASCADE'), nullable=False)
    folder_id     = Column(Integer, ForeignKey('folder.id', ondelete='SET NULL'), nullable=True)
    note_id       = Column(Integer, ForeignKey('note.id', ondelete='CASCADE'), nullable=True)
    original_name = Column(String(255), nullable=False)
    saved_path    = Column(String(512), nullable=False)
    content_type  = Column(String(100), nullable=False)
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))

    # ✅ 관계
    user   = relationship("User", back_populates="files")
    folder = relationship("Folder", back_populates="files")
    note   = relationship("Note", back_populates="files")
