from pydantic import BaseModel

class RegisterRequest(BaseModel):
    loginId: str
    email: str
    password: str

class RegisterResponse(BaseModel):
    message: str
    user_id: str

class LoginRequest(BaseModel):
    loginId: str
    password: str

class LoginResponse(BaseModel):
    message: str
    user_id: str

class KakaoRegisterRequest(BaseModel):
    access_token: str

class GoogleLoginRequest(BaseModel):
    token: str

class NaverLoginRequest(BaseModel):
    code: str
    state: str
