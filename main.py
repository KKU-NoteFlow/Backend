# src/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth import router as auth_router
from routers.note import router as note_router
from routers.folder import router as folder_router 
from fastapi.staticfiles import StaticFiles
from routers.file import router as file_router 
import logging




# 1) 환경변수 로드
load_dotenv()

# 2) 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3) FastAPI 앱 생성
app = FastAPI()

# 4) CORS 설정
origins = [
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 주소 (또는 "*"로 전체 허용 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5) 라우터 등록
app.include_router(auth_router)
app.include_router(note_router)
app.include_router(folder_router)  
app.include_router(file_router)

# 6) 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "mini"}

# 7) 실행 설정
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        env_file=".env"
    )
