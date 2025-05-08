# uvicorn main:app --host 0.0.0.0 --port 8081 --reload
from fastapi import FastAPI, HTTPException, APIRouter
from sqlalchemy import create_engine
import logging
# import jwt
# from jwt import PyJWTError
import uvicorn

from models import *
from routers import routers
from schemas import *

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 설정
app = FastAPI()

# 라우터 등록
for router in routers:
    app.include_router(router)


@app.get("/")
def read_root():
    return {"Hello": "World"}

# FastAPI 테스트 클라이언트 설정 
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)