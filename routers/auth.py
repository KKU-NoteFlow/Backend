from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db import get_db
from models import User
from schemas import RegisterRequest, RegisterResponse, LoginRequest, LoginResponse
import httpx, os
from services.kakao import get_kakao_user_info
from services.naver import get_naver_user_info
from services.google import get_google_user_info
from utils.jwt_utils import create_access_token

router = APIRouter(prefix="/api/v1", tags=["Auth"])

# 카카오 설정
KAKAO_TOKEN_URL    = "https://kauth.kakao.com/oauth/token"
KAKAO_CLIENT_ID    = os.getenv("KAKAO_CLIENT_ID")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")

# 네이버 설정
NAVER_TOKEN_URL     = "https://nid.naver.com/oauth2.0/token"
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_REDIRECT_URI  = os.getenv("NAVER_REDIRECT_URI")

# 구글 설정
GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
# 토큰 교환 시에도 프론트 콜백 주소 사용
GOOGLE_REDIRECT_URI  = os.getenv("GOOGLE_REDIRECT_URI")


@router.post("/register", response_model=RegisterResponse)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.id == request.loginId).first():
        raise HTTPException(400, "Id가 이미 존재합니다.")
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(400, "Email이 이미 존재합니다.")
    user = User(
        id=request.loginId,
        email=request.email,
        password=request.password,
        provider="local"
    )
    db.add(user); db.commit(); db.refresh(user)
    return RegisterResponse(message="User registered successfully", user_id=user.u_id)


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.loginId).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.password != request.password:
        raise HTTPException(401, "Invalid password")
    token = create_access_token({"sub": str(user.u_id)})
    return LoginResponse(message="Login successful", user_id=user.u_id, access_token=token)


@router.get("/auth/kakao/callback")
async def kakao_callback(code: str, db: Session = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            KAKAO_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": KAKAO_CLIENT_ID,
                "redirect_uri": KAKAO_REDIRECT_URI,
                "code": code
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
    if token_res.status_code != 200:
        raise HTTPException(400, "카카오 access_token 요청 실패")
    access_token = token_res.json().get("access_token")
    user_info = await get_kakao_user_info(access_token)
    kakao_id = str(user_info["id"])
    email = user_info["kakao_account"].get("email", f"{kakao_id}@kakao.local")
    user = db.query(User).filter(User.id == kakao_id, User.provider == "kakao").first()
    if not user:
        user = User(id=kakao_id, email=email, password="kakao_dummy", provider="kakao")
        db.add(user); db.commit(); db.refresh(user)
    jwt_token = create_access_token({"sub": str(user.u_id)})
    return {"access_token": jwt_token, "user_id": user.u_id}


@router.get("/auth/naver/callback")
async def naver_callback(code: str, db: Session = Depends(get_db)):
    params = {
        "grant_type":    "authorization_code",
        "client_id":     NAVER_CLIENT_ID,
        "client_secret": NAVER_CLIENT_SECRET,
        "code":          code
    }
    async with httpx.AsyncClient() as client:
        token_res = await client.get(NAVER_TOKEN_URL, params=params)
    if token_res.status_code != 200:
        raise HTTPException(400, "네이버 access_token 요청 실패")
    access_token = token_res.json().get("access_token")
    user_info = await get_naver_user_info(access_token)
    naver_id = str(user_info.get("id"))
    email = user_info.get("email", f"{naver_id}@naver.local")
    user = db.query(User).filter(User.id == naver_id, User.provider == "naver").first()
    if not user:
        user = User(id=naver_id, email=email, password="naver_dummy", provider="naver")
        db.add(user); db.commit(); db.refresh(user)
    jwt_token = create_access_token({"sub": str(user.u_id)})
    return {"access_token": jwt_token, "user_id": user.u_id}



@router.get("/auth/google/callback")
async def google_callback(code: str, db: Session = Depends(get_db)):
    # 1) access_token 교환
    data = {
        "grant_type":    "authorization_code",
        "client_id":     GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "code":          code,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    async with httpx.AsyncClient() as client:
        token_res = await client.post(GOOGLE_TOKEN_URL, data=data, headers=headers)
    if token_res.status_code != 200:
        detail = token_res.json()
        raise HTTPException(400, f"구글 access_token 요청 실패: {detail}")
    access_token = token_res.json()["access_token"]

    # 2) 사용자 정보 요청
    user_info = await get_google_user_info(access_token)
    # 긴 숫자 ID 대신 이메일 사용
    # google_user_id = str(user_info.get("id"))
    email          = user_info.get("email")
    login_id       = email            # <-- 여기를 email 로

    # 3) DB 조회/저장
    user = db.query(User).filter(User.id == login_id, User.provider == "google").first()
    if not user:
        user = User(
            id       = login_id,      # <-- 이메일을 id 로
            email    = email,
            password = "google_dummy",
            provider = "google"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # 4) JWT 발급 및 응답
    jwt_token = create_access_token(data={"sub": str(user.u_id)})
    return {"access_token": jwt_token, "user_id": user.u_id}

