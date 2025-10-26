# src/main.py
import os
from dotenv import load_dotenv
# 환경 변수를 최대한 빨리 로드하여 GPU 설정(CUDA_VISIBLE_DEVICES)이 라우터 임포트 전에 적용되도록 함
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth import router as auth_router
from routers.note import router as note_router
from routers.folder import router as folder_router 
from fastapi.staticfiles import StaticFiles
from routers.file import router as file_router 
import logging
import uvicorn  

# 1) 환경변수 로드 (상단에서 선 로드됨)

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
    allow_origins=["*"],
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
