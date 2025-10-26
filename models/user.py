from sqlalchemy import Column, Integer, String, Enum, TIMESTAMP, text
from sqlalchemy.orm import relationship
from .base import Base

class User(Base):
    __tablename__ = "user"

    u_id       = Column(Integer, primary_key=True, autoincrement=True)
    id         = Column(String(50),  nullable=False, unique=True)   # 로그인 ID 또는 소셜 ID
    email      = Column(String(150), nullable=False, unique=True)
    password   = Column(String(255), nullable=False)

    provider   = Column(
        Enum('local','google','kakao','naver', name='provider_enum'),
        nullable=False,
        server_default=text("'local'")
    )
    created_at = Column(TIMESTAMP, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(TIMESTAMP, nullable=False,
                        server_default=text('CURRENT_TIMESTAMP'),
                        onupdate=text('CURRENT_TIMESTAMP'))

    # ✅ 관계
    folders = relationship("Folder", back_populates="user", cascade="all, delete")
    notes   = relationship("Note", back_populates="user", cascade="all, delete")
    files   = relationship("File", back_populates="user", cascade="all, delete")

    provider   = Column(Enum("local", "google", "kakao", "naver", name="provider_enum"),
                        nullable=False, server_default=text("'local'"))
    created_at = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"),
                        onupdate=text("CURRENT_TIMESTAMP"))

    # relations
    notes      = relationship("Note", back_populates="user", cascade="all, delete-orphan")
    files      = relationship("File", back_populates="user", cascade="all, delete-orphan")
    checklists = relationship("Checklist", back_populates="user", cascade="all, delete-orphan")
