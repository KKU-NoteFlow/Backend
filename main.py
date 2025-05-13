import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# 1) 환경변수 로드
load_dotenv()

# 2) 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3) FastAPI 앱 생성
app = FastAPI()

# 4) CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 프론트 엔드 어디서든 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5) 라우터 등록
from routers import routers
for router in routers:
    app.include_router(router)

# 6) 루트 엔드포인트
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# 7) 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        env_file=".env"
    )
