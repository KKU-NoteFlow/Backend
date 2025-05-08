from sqlalchemy import Column, BigInteger, String, Enum, TIMESTAMP, text
from .base import Base

class User(Base):
    __tablename__ = 'User'
    
    u_id = Column(BigInteger, primary_key=True, autoincrement=True)
    id = Column(String(50), nullable=False, unique=True)
    email = Column(String(150), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    provider = Column(                                              
        Enum('local', 'google', 'kakao', 'naver', name='provider_enum'),
        nullable=False,
        server_default=text("'local'")
    )                                   
    created_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=text('CURRENT_TIMESTAMP')
    )                               
    updated_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP')
    )                                           