# model_server.py
"""
FastAPI 기반 모델 서버

- 기존 Final_Project의 LLM/VLM/RAG/상태머신 로직을 HTTP API로 감싼다.
- main.py, llm_core.py, input_vlm.py, rag_retriever.py, product_filter.py 등을 그대로 재사용.

엔드포인트
---------

1) POST /chat/text
   - 텍스트 기반 대화/추천
   - 요청: { "session_id": "optional", "message": "사용자 입력" }
   - 응답: {
       "session_id": "...",
       "reply": "LLM 응답",
       "mode": "SMALLTALK|SURVEY|RECOMMEND",
       "llm_latency": 1.23,
       "debug_state_summary": "선택적으로 상태 요약"
     }

2) POST /chat/image
   - 방/레퍼런스 이미지 업로드 → VLM 분석 + 상태 업데이트
   - form-data:
       - session_id: (optional) 문자열
       - is_want: (optional, default=false) "true"/"false"
       - file: 이미지 파일
   - 응답: {
       "session_id": "...",
       "message": "VLM 결과/안내 메시지",
       "debug_state_summary": "상태 요약 (텍스트)"
     }

3) POST /session/reset
   - 세션 전체 리셋 (상태 + 히스토리 삭제)
   - 요청: { "session_id": "..." }
   - 응답: { "session_id": "...", "status": "reset" }

로컬에서 테스트
---------------
가상환경(final_project)에서:

    uvicorn model_server:app --reload --host 0.0.0.0 --port 8000

"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading

# 기존 모듈들에서 필요한 것들 가져오기
from llm_core import parse_user_query  # 카테고리/무드/예산/공간 파싱

# main.py에는 상태머신과 모드별 핸들러가 들어있다고 가정
from main import (
    ChatState,
    ChatMode,
    decide_mode,
    handle_smalltalk,
    handle_survey,
    handle_recommend,
    render_summary,
    handle_image_command,
)

# ------------------------------------------------------------
# FastAPI 앱 및 CORS 설정
# ------------------------------------------------------------

app = FastAPI(
    title="Mood-based Interior Recommendation Model Server",
    description="Qwen2.5-14B-Korean + VLM + Chroma RAG 기반 모델 서버",
    version="0.1.0",
)

# 필요하면 개발 단계에서 CORS 완전 개방 (나중에 도메인 제한 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 세션 상태/히스토리 저장소 (간단한 in-memory 구현)
# ------------------------------------------------------------

_session_lock = threading.Lock()
_sessions: Dict[str, ChatState] = {}
_histories: Dict[str, List[Tuple[str, str]]] = {}

# 업로드 이미지 임시 저장 디렉토리
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp_uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _get_or_create_session(
    session_id: Optional[str],
) -> Tuple[str, ChatState, List[Tuple[str, str]]]:
    """
    세션 ID가 없으면 새로 만들고, 있으면 기존 상태/히스토리를 가져온다.
    """
    with _session_lock:
        if not session_id or session_id not in _sessions:
            new_id = session_id or str(uuid.uuid4())
            _sessions[new_id] = ChatState()
            _histories[new_id] = []
            return new_id, _sessions[new_id], _histories[new_id]

        return session_id, _sessions[session_id], _histories[session_id]


def _reset_session(session_id: str) -> None:
    """
    세션 상태와 히스토리를 완전히 삭제.
    """
    with _session_lock:
        _sessions.pop(session_id, None)
        _histories.pop(session_id, None)


# ------------------------------------------------------------
# Pydantic 요청/응답 모델
# ------------------------------------------------------------

class TextChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class TextChatResponse(BaseModel):
    session_id: str
    reply: str
    mode: str
    llm_latency: float
    debug_state_summary: Optional[str] = None


class SessionResetRequest(BaseModel):
    session_id: str


class SessionResetResponse(BaseModel):
    session_id: str
    status: str


class ImageChatResponse(BaseModel):
    session_id: str
    message: str
    debug_state_summary: Optional[str] = None


# ------------------------------------------------------------
# 헬스 체크
# ------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------------------------------------------------
# 텍스트 대화 엔드포인트
# ------------------------------------------------------------

@app.post("/chat/text", response_model=TextChatResponse)
def chat_text(req: TextChatRequest):
    """
    텍스트 기반 대화/추천 엔드포인트.

    - 기존 main.py의 한 턴 루프 로직을 그대로 옮겨온 구조:
      · parse_user_query()
      · decide_mode()
      · ChatMode에 따라 handle_smalltalk / handle_survey / handle_recommend 호출
    """
    # 1) 세션 상태 가져오기/생성
    session_id, state, history = _get_or_create_session(req.session_id)

    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message가 비어 있습니다.")

    # 2) 특수 명령 일부 지원 (원하면 프론트에서 직접 API를 나눠 써도 됨)
    # ::summary → 상태 요약만 돌려줌
    if user_text.startswith("::summary"):
        summary = render_summary(state)
        return TextChatResponse(
            session_id=session_id,
            reply=summary,
            mode="SUMMARY",
            llm_latency=0.0,
            debug_state_summary=summary,
        )

    # 여기서는 ::reset_all, ::reset_moods 는 별도 API로 처리하는 걸 권장하므로 생략

    # 3) 본격 대화 처리
    state.last_user_message = user_text  # main.py와 동일한 필드 사용 가정

    # 3-1) 이번 턴 파싱
    parsed = parse_user_query(user_text)

    # 3-2) 모드 결정
    mode = decide_mode(user_text, parsed, state)
    state.last_intent = mode.name

    # 3-3) SMALLTALK이 아닐 때만 state에 누적
    if mode != ChatMode.SMALLTALK:
        state.update_from_parsed(parsed)

    # 3-4) 모드별 응답 생성
    if mode == ChatMode.SMALLTALK:
        answer, llm_sec = handle_smalltalk(user_text, history)
    elif mode == ChatMode.SURVEY:
        answer, llm_sec = handle_survey(user_text, state, history)
    else:
        # RECOMMEND 또는 기타 → 추천 모드로 처리
        answer, llm_sec = handle_recommend(user_text, state, history)

    # 3-5) 히스토리 업데이트
    history.append((user_text, answer))

    # 3-6) 디버그용 상태 요약 (원하면 프론트에서 안 써도 됨)
    debug_summary = render_summary(state)

    return TextChatResponse(
        session_id=session_id,
        reply=answer,
        mode=mode.name,
        llm_latency=float(llm_sec),
        debug_state_summary=debug_summary,
    )


# ------------------------------------------------------------
# 이미지 기반 VLM 엔드포인트
# ------------------------------------------------------------

@app.post("/chat/image", response_model=ImageChatResponse)
async def chat_image(
    session_id: Optional[str] = Form(None),
    is_want: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    방/레퍼런스 이미지 업로드 → VLM 분석 → 상태 업데이트.

    - 기존 main.py의 handle_image_command()를 그대로 재사용한다.
    - handle_image_command()는 내부에서:
      · analyze_room_image() (input_vlm.py)
      · 인테리어/소품 필터링
      · ChatState의 current_* / target_image_* 필드 업데이트
      · 한국어 요약 문자열을 반환
    """
    # 1) 세션 상태 가져오기/생성
    session_id, state, _history = _get_or_create_session(session_id)

    # 2) 업로드 파일을 임시 디렉토리에 저장
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일 이름이 비어 있습니다.")

    suffix = Path(file.filename).suffix or ".jpg"
    tmp_path = TMP_DIR / f"{session_id}_{uuid.uuid4().hex}{suffix}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    # 3) 기존 CLI용 handle_image_command를 재사용하기 위해
    #    "-want" 옵션 포함한 인자 문자열을 만들어 준다.
    if is_want:
        # 레퍼런스/원하는 분위기 이미지
        arg = f'-want "{tmp_path}"'
    else:
        # 현재 방 이미지
        arg = f'"{tmp_path}"'

    # 4) VLM 처리 + ChatState 업데이트
    try:
        message = handle_image_command(arg, state)
    finally:
        # 사용 끝난 임시 파일 제거 (실패해도 크게 상관 없으므로 예외 무시)
        try:
            tmp_path.unlink(missing_ok=True)  # Python 3.8 이하면 exist_ok 처리 필요
        except TypeError:
            # Python <3.8 호환용
            if tmp_path.exists():
                tmp_path.unlink()

    # 5) 상태 요약까지 같이 반환 (프론트에서 디버그 탭에 보여줄 수 있음)
    summary = render_summary(state)

    return ImageChatResponse(
        session_id=session_id,
        message=message,
        debug_state_summary=summary,
    )


# ------------------------------------------------------------
# 세션 리셋 엔드포인트
# ------------------------------------------------------------

@app.post("/session/reset", response_model=SessionResetResponse)
def reset_session(req: SessionResetRequest):
    """
    세션 전체 리셋 (상태 + 히스토리 삭제).

    - 프론트에서 새로운 대화를 시작하고 싶을 때 호출.
    """
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id가 비어 있습니다.")

    _reset_session(req.session_id)
    return SessionResetResponse(session_id=req.session_id, status="reset")
