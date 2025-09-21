from sqlalchemy import Column, Integer, String, Enum, TIMESTAMP, text
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "user"

    u_id = Column(Integer, primary_key=True, autoincrement=True)  # PK
    id = Column(String(50), unique=True, nullable=False)          # 로그인 ID
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    provider = Column(
        Enum("local", "google", "kakao", "naver", name="provider_enum"),
        nullable=False,
        server_default="local",
    )
    created_at = Column(
        TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
    )

    # 역참조: checklist 목록
    checklists = relationship("Checklist", back_populates="user", cascade="all, delete-orphan")