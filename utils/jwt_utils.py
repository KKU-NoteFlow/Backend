# utils/jwt_utils.py

from datetime import datetime, timedelta, timezone
from typing import Optional            # Optional 임포트 추가
import jwt
import bcrypt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from models import User               # Member → User로 변경
from db import get_db

# JWT 설정값
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")

# 비밀번호 확인 함수
def verify_password(plain_password: str, hashed_password: str) -> bool:
    # bcrypt 해시 접두사 검사
    if hashed_password.startswith("$2a$") or hashed_password.startswith("$2b$"):
        return bcrypt.checkpw(plain_password.encode('utf-8'),
                              hashed_password.encode('utf-8'))
    # 해시 형식이 아닐 때 항상 False
    return False

# JWT 토큰 생성 함수
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    # sub 필드가 없으면 직접 추가 (권장)
    if "sub" not in to_encode:
        raise ValueError("`data` 딕셔너리에 반드시 'sub' 키가 포함되어야 합니다.")
    # 만료시간 계산
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    # 토큰 발급
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 현재 사용자 가져오기
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except (jwt.PyJWTError, ValueError):
        raise credentials_exception

    # Member → User로 변경
    user = db.query(User).filter(User.u_id == user_id).first()
    if user is None:
        raise credentials_exception
    return user
