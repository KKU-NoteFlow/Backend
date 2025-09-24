from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, text
from .base import Base

class File(Base):
    __tablename__ = 'file'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey('user.u_id', ondelete='CASCADE'), nullable=False)
    folder_id     = Column(Integer, ForeignKey('folder.id', ondelete='SET NULL'), nullable=True)
    note_id       = Column(Integer, ForeignKey('note.id', ondelete='SET NULL'), nullable=True)  # ✅ 첨부된 노트 ID
    original_name = Column(String(255), nullable=False)   # 유저가 업로드한 원본 파일 이름
    saved_path    = Column(String(512), nullable=False)   # 서버에 저장된(실제) 경로
    content_type  = Column(String(100), nullable=False)   # MIME 타입
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
