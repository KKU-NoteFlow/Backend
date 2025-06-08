# routers/auth.py

import os
import requests
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from db import get_db
from models.user import User
from schemas.user import (
    RegisterRequest, RegisterResponse,
    LoginRequest,    LoginResponse,
    GoogleLoginRequest, NaverLoginRequest
)
from utils.password import hash_password, verify_password
from utils.jwt_utils import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import logging

router = APIRouter(prefix="/api/v1", tags=["Auth"])

# Kakao 설정
KAKAO_CLIENT_ID    = os.getenv("KAKAO_CLIENT_ID")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")


@router.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # 중복 검사
    if db.query(User).filter(User.id == req.loginId).first():
        raise HTTPException(status_code=400, detail="Login ID already exists")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        id=req.loginId,
        email=req.email,
        password=hash_password(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RegisterResponse(
        message="User registered successfully",
        user_id=user.u_id
    )


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == req.loginId).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(req.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid password")

    # 토큰 생성 (expires_delta 없이 호출하면 환경변수에 설정된 만료 시간이 적용됩니다)
    access_token = create_access_token(user_id=user.u_id)

    return LoginResponse(
        message="Login successful",
        user_id=user.u_id,
        access_token=access_token,
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )


@router.post("/login/google", response_model=LoginResponse)
def login_google(req: GoogleLoginRequest, db: Session = Depends(get_db)):
    try:
        id_info = id_token.verify_oauth2_token(
            req.token,
            grequests.Request(),
            os.getenv("GOOGLE_CLIENT_ID")
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    google_id = id_info.get("sub")
    email     = id_info.get("email")

    user = db.query(User).filter(
        User.id == google_id,
        User.provider == 'google'
    ).first()
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

    access_token = create_access_token(user_id=user.u_id)
    return LoginResponse(
        message="Google login success",
        user_id=user.u_id,
        access_token=access_token,
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )


@router.post("/login/naver", response_model=LoginResponse)
def login_naver(req: NaverLoginRequest, db: Session = Depends(get_db)):
    # 1) 액세스 토큰 요청
    token_url = (
        f"https://nid.naver.com/oauth2.0/token"
        f"?grant_type=authorization_code"
        f"&client_id={os.getenv('NAVER_CLIENT_ID')}"
        f"&client_secret={os.getenv('NAVER_CLIENT_SECRET')}"
        f"&code={req.code}&state={req.state}"
    )
    token_res = requests.get(token_url)
    if token_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Naver token 요청 실패")

    access_token_val = token_res.json().get("access_token")
    headers         = {"Authorization": f"Bearer {access_token_val}"}

    # 2) 프로필 조회
    profile_res  = requests.get("https://openapi.naver.com/v1/nid/me", headers=headers)
    profile_data = profile_res.json()
    if profile_data.get("resultcode") != "00":
        raise HTTPException(status_code=400, detail="Naver 사용자 정보 요청 실패")

    naver_id = profile_data["response"]["id"]
    email    = profile_data["response"].get("email", f"{naver_id}@naver.local")

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

    access_token = create_access_token(user_id=user.u_id)
    return LoginResponse(
        message="Naver login success",
        user_id=user.u_id,
        access_token=access_token,
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )


@router.post("/auth/kakao/callback", response_model=LoginResponse)
def kakao_callback(code: str = Form(...), db: Session = Depends(get_db)):
    logging.warning(f"받은 code: {code}")

    token_url = "https://kauth.kakao.com/oauth/token"
    token_data = {
        "grant_type":    "authorization_code",
        "client_id":     KAKAO_CLIENT_ID,
        "redirect_uri":  KAKAO_REDIRECT_URI,
        "code":          code,
    }
    token_headers = { "Content-Type": "application/x-www-form-urlencoded" }

    token_res = requests.post(token_url, data=token_data, headers=token_headers)
    if token_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Kakao token 요청 실패")

    kakao_access_token = token_res.json().get("access_token")
    profile_res = requests.get(
        "https://kapi.kakao.com/v2/user/me",
        headers={"Authorization": f"Bearer {kakao_access_token}"}
    )
    if profile_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Kakao 사용자 정보 요청 실패")

    kakao_info    = profile_res.json()
    kakao_id      = str(kakao_info.get("id"))
    kakao_account = kakao_info.get("kakao_account", {})
    email         = kakao_account.get("email", f"{kakao_id}@kakao.local")

    user = db.query(User).filter(User.id == kakao_id, User.provider == "kakao").first()
    if not user:
        user = User(
            id=kakao_id,
            email=email,
            password="kakao_dummy",
            provider="kakao"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(user_id=user.u_id)
    return LoginResponse(
        message="Kakao login success",
        user_id=user.u_id,
        access_token=access_token,
        expires_in_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
