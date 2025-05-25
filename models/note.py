from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, TIMESTAMP, text
from .base import Base

class Note(Base):
    __tablename__ = 'note'

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(Integer, ForeignKey('user.u_id', ondelete='CASCADE'), nullable=False)
    folder_id     = Column(Integer, ForeignKey('folder.id', ondelete='SET NULL'), nullable=True)
    title         = Column(String(255), nullable=False)
    content       = Column(Text)
    is_favorite   = Column(Boolean, nullable=False, server_default=text('FALSE'))
    last_accessed = Column(TIMESTAMP, nullable=True)
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at    = Column(TIMESTAMP, nullable=False,
                           server_default=text('CURRENT_TIMESTAMP'),
                           onupdate=text('CURRENT_TIMESTAMP'))
