from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db import get_db
from models import User
from schemas import *

router = APIRouter(prefix="/api/v1", tags=["Auth"])
KAKAO_USER_INFO_URL = "https://kapi.kakao.com/v2/user/me"



@router.post("/register", response_model=RegisterResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # 중복 체크
    if db.query(User).filter(User.id == request.loginId).first():
        raise HTTPException(status_code=400, detail="Login ID already exists")
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # 사용자 저장
    user = User(
        id=request.loginId,
        email=request.email,
        password=request.password
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(message="User registered successfully", user_id=user.id)

# @router.post("/register/kakao")
# def register_kakao(request: KakaoRegisterRequest, db: Session = Depends(get_db)):
#     # 1. 카카오 사용자 정보 요청
#     headers = {"Authorization": f"Bearer {request.access_token}"}
#     response = requests.get(KAKAO_USER_INFO_URL, headers=headers)

#     if response.status_code != 200:
#         raise HTTPException(status_code=401, detail="Failed to fetch Kakao user info")

#     kakao_data = response.json()
#     kakao_id = kakao_data.get("id")  # 정수형, DB에 user.id로 저장
#     account_info = kakao_data.get("kakao_account", {})
#     properties = kakao_data.get("properties", {})

#     email = account_info.get("email", f"{kakao_id}@kakao.local")
#     name = properties.get("nickname", "KakaoUser")

#     # 2. 기존 사용자 확인
#     user = db.query(User).filter(User.id == kakao_id, User.provider == 'kakao').first()

#     if not user:
#         # 3. 새 사용자 등록
#         user = User(
#             id=kakao_id,  # 카카오 ID를 기본키로 저장
#             login_id=str(kakao_id),  # login_id에도 문자열 형태로 저장
#             email=email,
#             name=name,
#             password="kakao_dummy",  # 비밀번호는 사용하지 않음
#             provider="kakao"
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     return {
#         "message": "Kakao registration/login successful",
#         "user_id": user.id,
#         "provider": user.provider
#     }



@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # login_id에 해당하는 사용자 조회
    user = db.query(User).filter(User.id == request.loginId).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.password != request.password:
        raise HTTPException(status_code=401, detail="Invalid password")

    return LoginResponse(
        message="Login successful",
        user_id=user.id
    )

