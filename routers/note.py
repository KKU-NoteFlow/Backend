import os
import re
import json
import difflib
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db import get_db
from models.note import Note
from models.file import File as FileModel
from schemas.note import NoteCreate, NoteUpdate, NoteResponse, FavoriteUpdate, NoteFile
from utils.jwt_utils import get_current_user
from utils.llm import (
    stream_summary_with_langchain,
    _strip_top_level_h1_outside_code,
    _hf_generate_once,
    _system_prompt,
    count_slides,
    normalize_and_renumber_slides,
)

load_dotenv()

router = APIRouter(prefix="/api/v1", tags=["Notes"])
BASE_API_URL = os.getenv("BASE_API_URL", "http://localhost:8000")

HF_MAX_NEW_TOKENS_LONG = int(os.getenv("HF_MAX_NEW_TOKENS_LONG", "32000"))
HF_MAP_MAX_NEW_TOKENS = int(os.getenv("HF_MAP_MAX_NEW_TOKENS", "12000"))
ENSURE_COMPLETION_PASSES = int(os.getenv("ENSURE_COMPLETION_PASSES", "3"))
SLIDES_MIN = int(os.getenv("SUMMARY_SLIDES_MIN", "8"))
SLIDES_MAX = int(os.getenv("SUMMARY_SLIDES_MAX", "40"))
SUMMARY_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "12000"))
SUMMARY_CHUNK_OVERLAP = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "1200"))


# ─────────────────────────────────────────────
# 직렬화
# ─────────────────────────────────────────────
def serialize_note(db: Session, note: Note, base_url: str) -> NoteResponse:
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


# ─────────────────────────────────────────────
# 간단 추출 요약 (백업용)
# ─────────────────────────────────────────────
def _fallback_extractive_summary(text: str) -> str:
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
    if not s or not s.strip():
        return False
    low = s.lower()
    if ('## tl;dr' in low or '## 핵심' in low or '## 핵심 요점' in low) and len(s) > 300:
        return True
    headers = len(re.findall(r"^##\s+", s, flags=re.M))
    if headers >= 2 and len(s) > 200:
        return True
    return False


def _similarity_ratio(a: str, b: str) -> float:
    a_norm = re.sub(r"\s+", " ", (a or "")).strip()
    b_norm = re.sub(r"\s+", " ", (b or "")).strip()
    if not a_norm or not b_norm:
        return 0.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


async def _ensure_completion(full: str, domain: str | None = None, length: str = 'long') -> str:
    try:
        for _ in range(ENSURE_COMPLETION_PASSES):
            if _is_summary_complete(full) and re.search(r"[\.\!\?]\s*$", full.strip()):
                return full
            sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length=length)
            cont_prompt = (
                "The following summary appears incomplete. Continue and finish the summary **without repeating previous text**.\n\n"
                + full + "\n\nContinue:"
            )
            try:
                cont = await _hf_generate_once(sys_prompt, cont_prompt, max_new_tokens=HF_MAX_NEW_TOKENS_LONG)
            except Exception:
                cont = ''
            if cont and cont.strip():
                full = (full + "\n\n" + cont.strip()).strip()
            else:
                break
    except Exception:
        pass
    return full


async def _ensure_slide_coverage(full: str, target_slides: int, source_text: str, domain: str | None = None) -> str:
    try:
        for _ in range(ENSURE_COMPLETION_PASSES):
            cur = count_slides(full)
            if cur >= target_slides:
                return full

            next_idx = cur + 1
            sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length='long')
            cont_user = (
                "아래는 기존 요약입니다. '## 슬라이드' 섹션의 슬라이드 수가 목표보다 적습니다.\n"
                f"목표 슬라이드 수: {target_slides}\n"
                f"현재 슬라이드 수: {cur}\n\n"
                "요청: 이전 내용을 반복하지 말고, **'## 슬라이드' 섹션만** 이어서 작성하세요. "
                f"번호는 '### 슬라이드 {next_idx}'부터 연속으로 증가시키세요. "
                "각 슬라이드는 제목 + 3–6개 불릿로 작성하고, 아직 다루지 않은 원문 토픽을 중심으로 추가하세요.\n\n"
                "=== 기존 요약(참고) ===\n" + full[-12000:] + "\n\n"
                "=== 원문(발췌; 필요시) ===\n" + (source_text[:12000] if source_text else "")
            )
            try:
                extra = await _hf_generate_once(sys_prompt, cont_user, max_new_tokens=HF_MAX_NEW_TOKENS_LONG)
            except Exception:
                extra = ""

            if extra and extra.strip():
                full = (full.rstrip() + "\n\n" + extra.strip()).strip()
            else:
                break
    except Exception:
        pass
    return full


async def _force_compress_if_similar(full: str, source: str, domain: str | None = None) -> str:
    try:
        ratio = _similarity_ratio(full, source)
        if ratio >= 0.85 or len(full.strip()) >= max(300, int(len(source.strip()) * 0.95)):
            sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length='medium')
            user = (
                "다음 원문을 20–40% 길이로 정확하게 요약해. 절대 원문을 그대로 복사하지 말고, "
                "출력은 반드시 '## TL;DR', '## 핵심 요점', '## 상세 설명', '## 슬라이드' 섹션을 포함하라.\n\n"
                + (source[:80000] if source else "")
            )
            try:
                compressed = await _hf_generate_once(sys_prompt, user, max_new_tokens=HF_MAX_NEW_TOKENS_LONG)
            except Exception:
                compressed = _fallback_extractive_summary(source)
            if compressed and compressed.strip():
                return compressed
    except Exception:
        pass
    return full


# ─────────────────────────────────────────────
# 목록/CRUD
# ─────────────────────────────────────────────
@router.get("/notes", response_model=List[NoteResponse])
def list_notes(
    request: Request,
    q: Optional[str] = Query(default=None, description="Optional search query (title or content)"),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    query = db.query(Note).filter(Note.user_id == user.u_id)
    if q and q.strip():
        like = f"%{q.strip()}%"
        query = query.filter((Note.title.ilike(like)) | (Note.content.ilike(like)))

    notes = query.order_by(Note.created_at.desc()).all()
    base_url = os.getenv("BASE_API_URL") or str(request.base_url).rstrip('/')
    return [serialize_note(db, n, base_url) for n in notes]


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
# 요약 (로컬 Qwen 모델 기반, ChatGPT 스타일 자연요약)
# ─────────────────────────────────────────────
@router.post("/notes/{note_id}/summarize_sync", response_model=NoteResponse)
async def summarize_sync(
    note_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    """
    ✅ ChatGPT 스타일 요약 + 요약 완료 후 메모리 해제
    """
    import torch
    import numpy as np
    import gc
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    source = note.content.strip()
    if len(source) < 50:
        raise HTTPException(status_code=400, detail="본문이 너무 짧습니다.")

    full_summary = ""
    failed = False

    try:
        print("[summarize_sync] 🚀 Qwen2.5-7B-Instruct 로드 중...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 전문적인 과학기술 문서 요약가입니다. "
                    "텍스트를 자연스럽고 명확하게 요약하세요. "
                    "결과는 Markdown 형식으로 작성하고, 다음 구조를 유지하세요:\n\n"
                    "## 요약\n\n"
                    "## 핵심 요점\n\n"
                    "## 상세 설명\n"
                ),
            },
            {
                "role": "user",
                "content": f"아래 내용을 ChatGPT처럼 깔끔하고 자연스럽게 요약해줘:\n\n{source}",
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        print("[summarize_sync] 🧠 요약 생성 중...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1500, temperature=0.4, top_p=0.9)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        full_summary = generated.strip()

        print("[summarize_sync] ✅ 요약 완료")

    except Exception as e:
        print(f"[summarize_sync] ❌ 모델 요약 실패: {e}")
        failed = True

    finally:
        # ✅ 메모리 해제
        try:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[summarize_sync] 🧹 모델 메모리 해제 완료")
        except Exception as e:
            print(f"[summarize_sync] ⚠️ 메모리 해제 실패: {e}")

    # ───────────────
    # Fallback (TextRank)
    # ───────────────
    if failed or not full_summary:
        print("[summarize_sync] ⚠️ TextRank 백업 사용")
        try:
            sents = re.split(r"(?<=[.!?。])\s+|\n+", source)
            sents = [s.strip() for s in sents if len(s.strip()) > 10]
            if len(sents) < 3:
                full_summary = _fallback_extractive_summary(source)
            else:
                vec = TfidfVectorizer()
                tfidf = vec.fit_transform(sents)
                sim = cosine_similarity(tfidf)
                scores = np.sum(sim, axis=1)
                top_n = max(3, int(len(sents) * 0.2))
                top_idx = np.argsort(scores)[-top_n:]
                key_sents = [sents[i] for i in sorted(top_idx)]
                bullets = "\n".join(f"- {s}" for s in key_sents[:5])
                full_summary = f"## 요약\n{' '.join(key_sents[:2])}\n\n## 핵심 요점\n{bullets}\n\n## 상세 설명\n이 요약은 TextRank 기반 로컬 요약입니다."
        except Exception:
            full_summary = _fallback_extractive_summary(source)

    # ───────────────
    # DB 저장
    # ───────────────
    title = (note.title or "").strip() + " — 요약"
    if len(title) > 255:
        title = title[:255]

    new_note = Note(
        user_id=user.u_id,
        folder_id=note.folder_id,
        title=title,
        content=full_summary,
    )
    db.add(new_note)
    db.commit()
    db.refresh(new_note)

    base_url = os.getenv("BASE_API_URL") or "http://localhost:8000"
    return serialize_note(db, new_note, base_url)



# ─────────────────────────────────────────────
# 퀴즈 생성
# ─────────────────────────────────────────────
@router.post("/notes/{note_id}/generate-quiz")
def generate_quiz(
    note_id: int,
    count: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="퀴즈를 생성할 노트가 없습니다")

    text = (note.content or "").strip()
    import re as _re, random as _random
    sents = _re.split(r"(?<=[.!?。])\s+|\n+", text)
    sents = [s.strip() for s in sents if len(s.strip()) >= 8]
    _random.seed(note_id)
    _random.shuffle(sents)

    quizzes = []
    for s in sents:
        if len(quizzes) >= count:
            break
        toks = s.split()
        cand = [i for i, t in enumerate(toks) if len(_re.sub(r"\W+", "", t)) >= 4]
        if not cand:
            continue
        idx = cand[0]
        answer = _re.sub(r"^[\W_]+|[\W_]+$", "", toks[idx])
        toks[idx] = "_____"
        q = " ".join(toks)
        quizzes.append({
            "type": "cloze",
            "question": q,
            "answer": answer,
            "source": s,
        })

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
