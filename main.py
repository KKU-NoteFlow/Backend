# Backend/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from db import init_db
from routers.auth import router as auth_router
from routers.note import router as note_router
from routers.folder import router as folder_router
from routers.checklist import router as checklist_router
from routers.file import router as file_router

import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일(업로드) 서빙
os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "uploads")), name="static")

# 라우터 등록
app.include_router(auth_router)
app.include_router(note_router)
app.include_router(folder_router)
app.include_router(file_router)
app.include_router(checklist_router)

@app.get("/")
def root():
    return {"message": "mini"}

# 앱 시작 시(uvicorn main:app) 한 번만 테이블 생성 (개발용)
init_db()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
