from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from db import get_db
from models.user import User
from schemas.user import *
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import os
import requests

router = APIRouter(prefix="/api/v1", tags=["Auth"])

@router.post("/register", response_model=RegisterResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.id == request.loginId).first():
        raise HTTPException(status_code=400, detail="Login ID already exists")
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        id=request.loginId,
        email=request.email,
        password=request.password
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(message="User registered successfully", user_id=user.id)

@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.loginId).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.password != request.password:
        raise HTTPException(status_code=401, detail="Invalid password")

    return LoginResponse(
        message="Login successful",
        user_id=user.id
    )

@router.post("/login/google", response_model=LoginResponse)
def login_google(request: GoogleLoginRequest, db: Session = Depends(get_db)):
    token = request.token
    try:
        id_info = id_token.verify_oauth2_token(
            token,
            grequests.Request(),
            os.getenv("GOOGLE_CLIENT_ID")
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    google_id = id_info.get("sub")
    email = id_info.get("email")

    user = db.query(User).filter(User.id == google_id, User.provider == 'google').first()
    if not user:
        user = User(
            id=google_id,
            email=email,
            password="google_dummy",
            provider="google"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return LoginResponse(
        message="Google Login Success",
        user_id=user.id
    )

@router.post("/login/naver", response_model=LoginResponse)
def login_naver(request: NaverLoginRequest, db: Session = Depends(get_db)):
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    redirect_uri = "http://localhost:5173/naver/callback"

    # 1. access_token 요청
    token_url = (
        f"https://nid.naver.com/oauth2.0/token?grant_type=authorization_code"
        f"&client_id={client_id}&client_secret={client_secret}"
        f"&code={request.code}&state={request.state}"
    )
    token_res = requests.get(token_url)
    if token_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Naver token 요청 실패")

    token_data = token_res.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Naver access_token 없음")

    # 2. 사용자 정보 요청
    headers = { "Authorization": f"Bearer {access_token}" }
    profile_res = requests.get("https://openapi.naver.com/v1/nid/me", headers=headers)
    profile_data = profile_res.json()

    if profile_data.get("resultcode") != "00":
        raise HTTPException(status_code=400, detail="Naver 사용자 정보 요청 실패")

    naver_id = profile_data["response"]["id"]
    email = profile_data["response"].get("email", f"{naver_id}@naver.local")

    # 3. 기존 사용자 확인 및 저장
    user = db.query(User).filter(User.id == naver_id, User.provider == 'naver').first()
    if not user:
        user = User(
            id=naver_id,
            email=email,
            password="naver_dummy",
            provider="naver"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return LoginResponse(message="Naver login success", user_id=user.id)
