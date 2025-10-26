# Noteflow Backend (FastAPI)

## Overview
- FastAPI backend for Noteflow
- OCR pipeline supports images, PDF, DOC/DOCX, HWP (via utilities and system tools)

## Run (local)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Env (optional):
- `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`
- Database URLs if you connect a DB (current code uses provided models)

## OCR system tools (optional but recommended)
- PyMuPDF (Python) used by default for PDF text extraction
- Optional fallbacks/tools:
  - Poppler (`pdftoppm`) for `pdf2image`
  - LibreOffice (`soffice`) for .doc → .pdf
  - `hwp5txt` for .hwp text extraction
- If missing, the API still returns 200 with `warnings` explaining limitations.

## API Highlights
- `POST /api/v1/files/ocr` — OCR and create note (accepts file + optional `folder_id`, `langs`, `max_pages`)
- `POST /api/v1/files/upload` — Upload files to folder
- `POST /api/v1/files/audio` — STT from audio, create/append to note

## CI (GitHub Actions)
- This folder includes `.github/workflows/ci.yml` to lint/smoke-test on push/PR.
- Python 3.11, `pip install -r requirements.txt`, syntax check and import smoke.

## Docker (optional; for later)
- Dockerfile included. Build & run locally:
```
docker build -t noteflow-backend .
docker run --rm -p 8080:8080 noteflow-backend
```
- GitHub Actions container build:
  - `.github/workflows/docker.yml` pushes to GHCR:
    - `ghcr.io/<owner>/<repo>:backend-latest`
    - `ghcr.io/<owner>/<repo>:backend-<sha>`
- Deployment example (SSH) once you’re ready:
```
docker login ghcr.io -u <USER> -p <TOKEN>
docker pull ghcr.io/<owner>/<repo>:backend-latest
docker run -d --name backend --restart=always -p 8080:8080 ghcr.io/<owner>/<repo>:backend-latest
```

## Notes
- If you split this folder into its own repository root, the included `.github/workflows/*.yml` files will work as-is.
- OCR uses model-first path (EasyOCR + TrOCR) and falls back to tesseract when available.
