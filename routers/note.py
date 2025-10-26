import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import traceback
import re
import json

from db import get_db, SessionLocal
from models.note import Note
from models.file import File as FileModel
from schemas.note import NoteCreate, NoteUpdate, NoteResponse, FavoriteUpdate, NoteFile
from utils.jwt_utils import get_current_user
from utils.llm import stream_summary_with_langchain, _strip_top_level_h1_outside_code, _hf_generate_once, _system_prompt
from utils.llm import _hf_generate_once, _system_prompt

load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")

router = APIRouter(prefix="/api/v1", tags=["Notes"])

# 환경변수에서 BASE_API_URL 가져와 파일 다운로드 URL 구성
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# 공통: Note → NoteResponse 직렬화 + files 채우기
# ─────────────────────────────────────────────
def serialize_note(db: Session, note: Note, base_url: str) -> NoteResponse:
    """
    Note ORM → NoteResponse 수동 매핑.
    관계(note.files)로 인해 Pydantic가 ORM 객체를 바로 검증하려다 실패하는 문제를 피하기 위해
    기본 스칼라 필드만 직접 채우고, files는 별도 쿼리로 구성한다.
    """
    files = (
        db.query(FileModel)
        .filter(FileModel.note_id == note.id, FileModel.user_id == note.user_id)
        .order_by(FileModel.created_at.desc())
        .all()
    )
    file_items: List[NoteFile] = [
        NoteFile(
            file_id=f.id,
            original_name=f.original_name,
            content_type=f.content_type,
            url=f"{base_url}/api/v1/files/download/{f.id}",
            created_at=f.created_at,
        )
        for f in files
    ]

    return NoteResponse(
        id=note.id,
        user_id=note.user_id,
        folder_id=note.folder_id,
        title=note.title,
        content=note.content,
        is_favorite=bool(note.is_favorite),
        last_accessed=note.last_accessed,
        created_at=note.created_at,
        updated_at=note.updated_at,
        files=file_items,
    )


def _fallback_extractive_summary(text: str) -> str:
    """Simple extractive fallback: pick leading sentences and format as TL;DR + bullets."""
    if not text:
        return "## TL;DR\n요약할 내용이 없습니다."
    sents = re.split(r"(?<=[.!?。])\s+|\n+", text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return "## TL;DR\n요약할 내용이 없습니다."
    tl = sents[0][:400]
    bullets = []
    for s in sents[1:6]:
        short = s[:200]
        bullets.append(f"- {short}")
    body = "\n\n## 핵심 요점\n" + "\n".join(bullets) if bullets else ""
    return f"## TL;DR\n{tl}{body}"


def _is_summary_complete(s: str) -> bool:
    """Heuristic: check presence of key sections and reasonable length."""
    if not s or not s.strip():
        return False
    low = s.lower()
    # require TL;DR or 핵심 요점 and some detail
    if ('## tl;dr' in low or '## 핵심' in low or '## 핵심 요점' in low) and len(s) > 300:
        return True
    # if contains multiple section headers, consider complete
    headers = len(re.findall(r"^##\s+", s, flags=re.M))
    if headers >= 2 and len(s) > 200:
        return True
    # otherwise likely incomplete
    return False


async def _ensure_completion(full: str, domain: str | None = None, length: str = 'long') -> str:
    """If `full` looks truncated, attempt up to 3 continuation passes to complete it."""
    try:
        for i in range(3):
            if _is_summary_complete(full) and re.search(r"[\.\!\?]\s*$", full.strip()):
                return full
            # build continuation prompt
            sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length=length)
            cont_prompt = "The following summary appears incomplete. Continue and finish the summary without repeating previous text:\n\n" + full + "\n\nContinue:" 
            try:
                cont = await _hf_generate_once(sys_prompt, cont_prompt, max_new_tokens=int(os.getenv('HF_MAX_NEW_TOKENS_LONG', '32000')))
            except Exception:
                cont = ''
            if cont and cont.strip():
                # append continuation
                full = (full + "\n\n" + cont.strip()).strip()
            else:
                break
    except Exception:
        pass
    return full


# 1) 모든 노트 조회
@router.get("/notes", response_model=List[NoteResponse])
def list_notes(
    request: Request,
    q: str | None = Query(default=None, description="Optional search query (title or content)"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """List notes for the current user. If `q` is provided, filter by title or content (case-insensitive).
    """
    query = db.query(Note).filter(Note.user_id == user.u_id)
    if q and q.strip():
        like = f"%{q.strip()}%"
        query = query.filter((Note.title.ilike(like)) | (Note.content.ilike(like)))

    notes = query.order_by(Note.created_at.desc()).all()
    # 각 노트의 files도 채워 반환
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return [serialize_note(db, n, base_url) for n in notes]


# 2) 최근 접근한 노트 조회 (상위 10개)
@router.get("/notes/recent", response_model=List[NoteResponse])
def recent_notes(
    request: Request,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    notes = (
        db.query(Note)
        .filter(Note.user_id == user.u_id, Note.last_accessed.isnot(None))
        .order_by(Note.last_accessed.desc())
        .limit(10)
        .all()
    )
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return [serialize_note(db, n, base_url) for n in notes]


# 3) 노트 생성
@router.post("/notes", response_model=NoteResponse)
def create_note(
    request: Request,
    req: NoteCreate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = Note(
        user_id=user.u_id,
        folder_id=req.folder_id,
        title=req.title,
        content=req.content
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return serialize_note(db, note, base_url)


# 4) 노트 수정 (제목/내용/폴더)
@router.patch("/notes/{note_id}", response_model=NoteResponse)
def update_note(
    request: Request,
    note_id: int,
    req: NoteUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    if req.title is not None:
        note.title = req.title
    if req.content is not None:
        note.content = req.content
    if req.folder_id is not None:
        note.folder_id = req.folder_id
    if req.is_favorite is not None:
        note.is_favorite = req.is_favorite

    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return serialize_note(db, note, base_url)


# 5) 노트 단일 조회 (마지막 접근 시간 업데이트 포함)
@router.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(
    request: Request,
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.last_accessed = datetime.utcnow()
    db.commit()
    db.refresh(note)
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return serialize_note(db, note, base_url)


# 6) 노트 삭제
@router.delete("/notes/{note_id}")
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    db.delete(note)
    db.commit()
    return {"message": "Note deleted successfully"}


# 7) 즐겨찾기 토글
@router.patch("/notes/{note_id}/favorite", response_model=NoteResponse)
def toggle_favorite(
    request: Request,
    note_id: int,
    req: FavoriteUpdate,
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note.is_favorite = req.is_favorite
    note.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(note)
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return serialize_note(db, note, base_url)


# ─────────────────────────────────────────────
# (참고) 요약 스트리밍 API - 완료 후에도 serialize_note 사용 안 함
#        (요약은 새 노트를 생성하고 SSE로 알림만 보냄)
# ─────────────────────────────────────────────
@router.post("/notes/{note_id}/summarize")
async def summarize_stream_langchain(
    note_id: int,
    background_tasks: BackgroundTasks,
    domain: str | None = Query(default=None, description="meeting | code | paper | general | auto(None)"),
    longdoc: bool = Query(default=True, description="Enable long-document map→reduce"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    

    async def event_gen():
        parts = []
        # Default to a comprehensive (long) summary when called without explicit options
        async for sse in stream_summary_with_langchain(note.content, domain=domain, longdoc=longdoc, length='long', tone='neutral', output_format='md'):
            parts.append(sse.removeprefix("data: ").strip())
            yield sse.encode()
        full = "".join(parts).strip()
        # attempt to complete if truncated
        try:
            full = await _ensure_completion(full, domain=domain, length='long')
        except Exception:
            pass
        # If streamed output looks incomplete, attempt a single-shot completion pass
        try:
            if not _is_summary_complete(full):
                try:
                    print('[summarize] partial output detected, performing completion pass')
                    sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length='long')
                    cont = await _hf_generate_once(sys_prompt, "Existing partial summary:\n\n" + full + "\n\nPlease expand and complete the summary, preserving facts and following the output format.", max_new_tokens=int(os.getenv('HF_MAX_NEW_TOKENS_LONG', '20000')))
                    if cont and cont.strip():
                        full = (full + "\n\n" + cont.strip()).strip()
                        print('[summarize] completion pass appended, new length=', len(full))
                except Exception as e:
                    print('[summarize] completion pass failed:', e)
        except Exception:
            pass
        # If model produced empty output, fall back to a simple extractive summary
        if not (full or "").strip():
            try:
                sents = re.split(r"(?<=[.!?。])\s+|\n+", note.content or "")
                sents = [p.strip() for p in sents if p.strip()]
                head = sents[:6]
                tl = head[0] if head else (note.content or "")[:200]
                bullets = [f"- {p}" for p in head[1:5]]
                fb = "## TL;DR\n" + tl + "\n\n## 핵심 요점\n" + "\n".join(bullets)
                full = fb
            except Exception:
                full = (note.content or "")[:800]
        try:
            print(f"[summarize-sync] generated full length={len(full)} preview={repr(full[:200])}")
        except Exception:
            pass
        # Remove local temp file paths (e.g. macOS /var/... or file://...) which shouldn't be persisted
        try:
            # remove explicit file://... patterns
            full = re.sub(r"file://\S+", "", full)
            # remove absolute tmp paths like /var/... (up to whitespace or closing paren)
            full = re.sub(r"/var/[^\s)]+", "", full)
            # remove parenthesis-wrapped local paths in markdown images: ![alt](/path/to/file.png)
            full = re.sub(r"!\[([^\]]*)\]\([^)]*(/var/[^)\s]+)[)]", r"![\1]()", full)
        except Exception:
            pass
        # Strip any top-level H1 headings that the model may have added (outside code fences)
        try:
            full = _strip_top_level_h1_outside_code(full)
        except Exception:
            # fallback: naive removal of a single leading H1
            full = re.sub(r"^\s*#\s.*?\n+", "", full, count=1)
        # Ensure non-empty summary; if model produced nothing, use extractive fallback
        if not (full or "").strip():
            try:
                full = _fallback_extractive_summary(note.content)
                print(f"[summarize] fallback summary used length={len(full)}")
            except Exception:
                full = (note.content or '')[:800]

        # Ensure non-empty summary; if model produced nothing, use extractive fallback
        if not (full or "").strip():
            try:
                full = _fallback_extractive_summary(note.content)
                print(f"[summarize-sync] fallback summary used length={len(full)}")
            except Exception:
                full = (note.content or '')[:800]

        if full:
            # Create a new summary note in the same folder with title '<original> — 요약'
            title = (note.title or "").strip() + " — 요약"
            if len(title) > 255:
                title = title[:255]
            new_note = Note(
                user_id=user.u_id,
                folder_id=note.folder_id,
                title=title,
                content=full,
            )
            db.add(new_note)
            db.commit()
            db.refresh(new_note)
            try:
                # log created summary id and content preview for debugging
                print(f"[summarize] created summary note id={new_note.id} for note_id={note_id}")
                try:
                    print("[summarize] saved content length=", len(new_note.content or ""))
                    print("[summarize] saved content preview=", repr((new_note.content or "")[:400]))
                except Exception:
                    pass
            except Exception:
                pass
            # normal streaming path: notify created note via SSE
            try:
                # notify created note: include serialized note JSON so client can render immediately
                base_url = os.getenv("BASE_API_URL") or BASE_API_URL
                note_obj = serialize_note(db, new_note, base_url)
                payload = {"summary_note": note_obj.dict()}
                yield f"data: {json.dumps(payload, default=str)}\n\n".encode()
            except Exception:
                # fallback to ID-only message
                try:
                    yield f"data: SUMMARY_NOTE_ID:{new_note.id}\n\n".encode()
                except Exception:
                    pass
        else:
            # As an extra fallback, aggregate streamed parts (if any) to ensure coverage
            try:
                agg = "\n\n".join(parts) if parts else (note.content or '')[:4000]
                fallback_full = "## Aggregated streamed parts\n\n" + agg
                title = (note.title or "").strip() + " — 요약"
                new_note2 = Note(user_id=user.u_id, folder_id=note.folder_id, title=title, content=fallback_full)
                db.add(new_note2)
                db.commit()
                db.refresh(new_note2)
                try:
                    yield f"data: SUMMARY_NOTE_ID:{new_note2.id}\n\n".encode()
                except Exception:
                    pass
            except Exception:
                pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )



@router.post("/notes/{note_id}/generate-quiz")
def generate_quiz(
    note_id: int,
    count: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    """간단한 규칙 기반 퀴즈 생성(대형 모델 없이 동작)."""
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="퀴즈를 생성할 노트가 없습니다")

    text = (note.content or "").strip()
    # 문장 단위 분할
    import re, random
    sents = re.split(r"(?<=[.!?。])\s+|\n+", text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 8]
    random.seed(note_id)
    random.shuffle(sents)

    quizzes = []
    for s in sents:
        if len(quizzes) >= count:
            break
        # 공백 기준 토큰화 후, 길이 4 이상인 토큰을 빈칸으로
        toks = s.split()
        cand = [i for i, t in enumerate(toks) if len(re.sub(r"\W+", "", t)) >= 4]
        if not cand:
            continue
        idx = cand[0]
        answer = re.sub(r"^[\W_]+|[\W_]+$", "", toks[idx])
        toks[idx] = "_____"
        q = " ".join(toks)
        quizzes.append({
            "type": "cloze",
            "question": q,
            "answer": answer,
            "source": s,
        })

    # 보강: 부족하면 참/거짓 생성
    i = 0
    while len(quizzes) < count and i < len(sents):
        stmt = sents[i]
        i += 1
        if len(stmt) < 12:
            continue
        false_stmt = stmt.replace("이다", "아니다").replace("다.", "가 아니다.")
        quizzes.append({
            "type": "boolean",
            "question": stmt,
            "answer": True,
        })
        if len(quizzes) >= count:
            break
        quizzes.append({
            "type": "boolean",
            "question": false_stmt,
            "answer": False,
        })

    return {"note_id": note.id, "count": len(quizzes), "items": quizzes}


# Convenience synchronous summarization endpoint (returns created note JSON).
@router.post("/notes/{note_id}/summarize_sync", response_model=NoteResponse)
async def summarize_sync(
    note_id: int,
    domain: str | None = Query(default=None, description="meeting | code | paper | general | auto(None)"),
    longdoc: bool = Query(default=True, description="Enable long-document map→reduce"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    parts = []
    async for sse in stream_summary_with_langchain(note.content, domain=domain, longdoc=longdoc, length='long', tone='neutral', output_format='md'):
        parts.append(sse.removeprefix("data: ").strip())
    full = "".join(parts).strip()

    # sanitize local paths and strip top-level H1
    try:
        full = re.sub(r"file://\S+", "", full)
        full = re.sub(r"/var/[^\s)]+", "", full)
        full = _strip_top_level_h1_outside_code(full)
    except Exception:
        try:
            full = re.sub(r"^\s*#\s.*?\n+", "", full, count=1)
        except Exception:
            pass

    # If model produced empty output, use extractive fallback
    if not (full or "").strip():
        try:
            full = _fallback_extractive_summary(note.content)
            print(f"[summarize_sync] fallback used length={len(full)}")
        except Exception:
            full = (note.content or '')[:800]

    title = (note.title or "").strip() + " — 요약"
    if len(title) > 255:
        title = title[:255]
    new_note = Note(
        user_id=user.u_id,
        folder_id=note.folder_id,
        title=title,
        content=full,
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)
    try:
        print(f"[summarize_sync] created summary note id={new_note.id} for note_id={note_id}")
        print("[summarize_sync] saved content length=", len(new_note.content or ""))
        print("[summarize_sync] saved content preview=", repr((new_note.content or "")[:400]))
    except Exception:
        pass
    base_url = os.getenv("BASE_API_URL") or "http://localhost:8000"
    return serialize_note(db, new_note, base_url)
