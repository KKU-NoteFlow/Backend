# schemas/user.py

from pydantic import BaseModel

class RegisterRequest(BaseModel):
    loginId: str
    email: str
    password: str

class RegisterResponse(BaseModel):
    message: str
    user_id: int               # int 타입으로 변경

class LoginRequest(BaseModel):
    loginId: str
    password: str 

class LoginResponse(BaseModel):
    message: str
    user_id: int               # int 타입으로 변경
    access_token: str

class KakaoLoginRequest(BaseModel):
    code: str

class GoogleLoginRequest(BaseModel):
    token: str

class NaverLoginRequest(BaseModel):
    code: str
    state: str
