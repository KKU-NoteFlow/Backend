from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from db import get_db
from models.user import User
from schemas.user import *
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
import os

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
