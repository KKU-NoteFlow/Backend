# NoteFlow Backend

[![Backend-FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![DB-MySQL](https://img.shields.io/badge/DB-MySQL-4479A1?logo=mysql&logoColor=white)](#)
[![ORM-SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-6E7781)](#)
[![Auth-JWT%20%26%20OAuth](https://img.shields.io/badge/Auth-JWT%20%26%20OAuth-4B5563)](#)
[![AI-OCR%20%26%20Whisper](https://img.shields.io/badge/AI-OCR%20%26%20Whisper-8A2BE2)](#)
[![기간](https://img.shields.io/badge/기간-2025.04–2025.10-6E7781)](#)

노트/폴더/파일을 중심으로 한 지식 관리 백엔드입니다. OCR(PDF/이미지/문서), 음성(STT), 요약/문제생성 등의 생산성 기능과 소셜 로그인(구글/네이버/카카오), JWT 인증을 제공합니다.

---

## 목차
- [기술 스택](#기술-스택)
- [아키텍처](#아키텍처)
- [데이터 모델(ERD)](#데이터-모델erd)
- [핵심 API](#핵심-api)
- [로컬 실행](#로컬-실행)
- [Docker](#docker)
- [폴더 구조](#폴더-구조)

---

## 기술 스택
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-6E7781?style=for-the-badge)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)

![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![Google OAuth](https://img.shields.io/badge/Google%20OAuth-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Naver/Kakao OAuth](https://img.shields.io/badge/Naver%20%7C%20Kakao-03C75A%20%7C%20F7E600?style=for-the-badge)

![OCR](https://img.shields.io/badge/OCR-PyMuPDF%20%7C%20pdf2image%20%7C%20pytesseract-4C9?style=for-the-badge)
![STT](https://img.shields.io/badge/STT-OpenAI%20Whisper-8A2BE2?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-HuggingFace%20%7C%20LangChain-ffcc00?style=for-the-badge)

---

## 아키텍처

```mermaid
flowchart LR
  client[Client]
  api[FastAPI]
  db[(MySQL)]
  oauth_g[Google]
  oauth_n[Naver]
  oauth_k[Kakao]
  svc[OCR & Whisper]

  client --> api
  api --> db
  client -.-> oauth_g
  client -.-> oauth_n
  client -.-> oauth_k
  api --> svc
```

---

## 데이터 모델(ERD)

```mermaid
erDiagram
    user ||--o{ folder : owns
    user ||--o{ note : writes
    user ||--o{ file : uploads
    user ||--o{ checklist : has
    folder ||--o{ note : contains
    folder ||--o{ file : contains
    note ||--o{ file : attaches

    user {
        int u_id PK
        string id
        string email
        string password
        enum provider
        datetime created_at
        datetime updated_at
    }

    folder {
        int id PK
        int user_id FK
        string name
        int parent_id
        datetime created_at
        datetime updated_at
    }

    note {
        int id PK
        int user_id FK
        int folder_id FK
        string title
        text content
        bool is_favorite
        datetime last_accessed
        datetime created_at
        datetime updated_at
    }

    file {
        int id PK
        int user_id FK
        int folder_id FK
        int note_id FK
        string original_name
        string saved_path
        string content_type
        datetime created_at
    }

    checklist {
        int id PK
        int user_id FK
        string title
        bool is_clear
    }
```

---

## 핵심 API

- 인증/회원가입: `POST /api/v1/register`, `POST /api/v1/login`, `POST /api/v1/login/google`, `POST /api/v1/login/naver`, `POST /api/v1/auth/kakao/callback`
- 노트: `GET/POST/PATCH/DELETE /api/v1/notes`, `GET /api/v1/notes/recent`, `PATCH /api/v1/notes/{id}/favorite`
- 요약/문제생성: `POST /api/v1/notes/{id}/summarize`, `POST /api/v1/notes/{id}/summarize_sync`, `POST /api/v1/notes/{id}/generate-quiz`
- 폴더: `GET/POST/PATCH/DELETE /api/v1/folders` (트리+노트 포함 응답)
- 파일: `POST /api/v1/files/upload`, `GET /api/v1/files/download/{file_id}`, `GET /api/v1/files/list/{folder_id}`
- OCR/STT: `POST /api/v1/files/ocr?langs=kor+eng&max_pages=50`, `POST /api/v1/files/audio`

토큰
- 모든 보호 API는 `Authorization: Bearer <JWT>` 헤더 요구

---

## 로컬 실행

### 1) 요구사항
- Python 3.10+
- MySQL 8.x (또는 호환 RDS)

### 2) 의존성 설치
```bash
cd Backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) 환경 변수 (.env)
```env
DATABASE_URL=mysql+mysqlconnector://USER:PASSWORD@localhost:3306/noteflow?charset=utf8mb4
SECRET_KEY=CHANGE_ME
ACCESS_TOKEN_EXPIRE_MINUTES=120
BASE_API_URL=http://localhost:8080

# OAuth
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
NAVER_CLIENT_ID=YOUR_NAVER_CLIENT_ID
NAVER_CLIENT_SECRET=YOUR_NAVER_CLIENT_SECRET
KAKAO_CLIENT_ID=YOUR_KAKAO_REST_API_KEY
KAKAO_REDIRECT_URI=http://localhost:5174/kakao/callback

# LLM (선택)
HF_API_TOKEN=YOUR_HF_TOKEN
HF_MAX_NEW_TOKENS_LONG=32000
```

### 4) 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

업로드 정적 경로
- 업로드 파일은 `Backend/uploads/`에 저장, 정적 서빙은 `/static`으로 노출됩니다.

OCR 의존 도구(선택)
- Poppler(`pdftoppm`), LibreOffice(`soffice`), `hwp5txt`가 있으면 더 폭넓은 포맷 지원
- 진단: `GET /api/v1/files/ocr/diag`

---

## Docker

```bash
cd Backend
docker build -t noteflow-backend .
docker run --rm -p 8080:8080 --env-file .env noteflow-backend
```

주의
- OCR/Whisper 등은 컨테이너에 추가 시스템 패키지가 필요할 수 있습니다.

---

## 폴더 구조

```
Backend/
├── main.py                # FastAPI 엔트리포인트(CORS, 라우터, 정적)
├── db.py                  # DB 연결/세션/초기화
├── models/                # SQLAlchemy 모델(User/Folder/Note/File/Checklist)
├── routers/               # auth, note, note_sync, folder, file, checklist
├── schemas/               # Pydantic 요청/응답 스키마
├── utils/                 # JWT/패스워드/LLM/OCR 유틸
├── uploads/               # 업로드 저장소(정적 서빙)
├── requirements.txt       # 의존성 목록
├── Dockerfile             # 컨테이너 빌드 스펙
└── Makefile               # 자주 쓰는 명령(선택)
```
