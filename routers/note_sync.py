from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
import os, re

from db import get_db
from models.note import Note
from schemas.note import NoteResponse
from utils.jwt_utils import get_current_user
from utils.llm import (
    stream_summary_with_langchain,
    _strip_top_level_h1_outside_code,
    _hf_generate_once,
    _system_prompt,
    _chunk_text,
    _compose_user_payload,
)

router = APIRouter(prefix="/api/v1", tags=["Notes"])


@router.post("/notes/{note_id}/summarize_sync", response_model=NoteResponse)
async def summarize_sync(
    note_id: int,
    domain: Optional[str] = Query(default=None, description="meeting | code | paper | general | auto(None)"),
    longdoc: bool = Query(default=True),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    note = db.query(Note).filter(Note.id == note_id, Note.user_id == user.u_id).first()
    if not note or not (note.content or "").strip():
        raise HTTPException(status_code=404, detail="요약 대상 없음")

    # Use map->reduce single-shot generation to avoid streaming truncation.
    # 1) chunk text
    chunks = _chunk_text(note.content or "", chunk_chars=int(os.getenv('SUMMARY_CHUNK_CHARS','20000')), overlap=int(os.getenv('SUMMARY_CHUNK_OVERLAP','2000')))
    map_sys = _system_prompt(domain or 'general', phase='map', output_format='md', length='long')
    partials = []
    for idx, ch in enumerate(chunks, 1):
        try:
            map_input = _compose_user_payload(ch, "", "md", length='short', tone='neutral')
            part = await _hf_generate_once(map_sys, map_input, max_new_tokens=int(os.getenv('HF_MAP_MAX_NEW_TOKENS','12000')))
        except Exception:
            part = (ch or '')[:800]
        partials.append(f"[Chunk {idx}]\n{part.strip()}")

    reduce_text = "\n\n".join(partials)
    reduce_sys = _system_prompt(domain or 'general', phase='reduce', output_format='md', length='long')
    reduce_input = _compose_user_payload(reduce_text, "", "md", length='long', tone='neutral')
    try:
        full = await _hf_generate_once(reduce_sys, reduce_input, max_new_tokens=int(os.getenv('HF_MAX_NEW_TOKENS_LONG','32000')))
    except Exception:
        full = (note.content or '')[:4000]
    # If partial/short, try a completion pass
    try:
        # local completeness heuristic
        def is_complete(s: str) -> bool:
            if not s or not s.strip():
                return False
            low = s.lower()
            if ('## tl;dr' in low or '## 핵심' in low or '## 핵심 요점' in low) and len(s) > 300:
                return True
            headers = len(re.findall(r"^##\s+", s, flags=re.M))
            return headers >= 2 and len(s) > 200

        if not is_complete(full):
            sys_prompt = _system_prompt(domain or 'general', phase='final', output_format='md', length='long')
            cont = await _hf_generate_once(sys_prompt, "Existing partial summary:\n\n" + (full or "") + "\n\nPlease expand and complete the summary, preserving facts and following the output format.", max_new_tokens=int(os.getenv('HF_MAX_NEW_TOKENS_LONG', '20000')))
            if cont and cont.strip():
                full = (full + "\n\n" + cont.strip()).strip()
    except Exception:
        pass
    # sanitize
    try:
        full = re.sub(r"file://\S+", "", full)
        full = re.sub(r"/var/[^\s)]+", "", full)
    except Exception:
        pass
    try:
        full = _strip_top_level_h1_outside_code(full)
    except Exception:
        full = re.sub(r"^\s*#\s.*?\n+", "", full, count=1)

    # If still incomplete or suspiciously short, fall back to aggregated chunk summaries to ensure coverage
    try:
        if not _is_summary_complete(full) or len(full) < 300:
            agg = "\n\n".join(partials)
            full = "## Aggregated chunk summaries\n\n" + agg
            print('[summarize_sync] using aggregated chunk summaries, length=', len(full))
    except Exception:
        pass

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
    base_url = os.getenv("BASE_API_URL") or "http://localhost:8000"
    return NoteResponse(
        id=new_note.id,
        user_id=new_note.user_id,
        folder_id=new_note.folder_id,
        title=new_note.title,
        content=new_note.content,
        is_favorite=bool(new_note.is_favorite),
        last_accessed=new_note.last_accessed,
        created_at=new_note.created_at,
        updated_at=new_note.updated_at,
        files=[],
    )
