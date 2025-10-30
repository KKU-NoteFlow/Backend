# schemas/__init__.py

from .user import (
    RegisterRequest, RegisterResponse,
    LoginRequest,    LoginResponse,
    GoogleLoginRequest, NaverLoginRequest,
    KakaoLoginRequest,
)

from .qg_schema import (
    QuestionItem,
    QuestionGenerationRequest,
    QuestionGenerationResponse,
)
 