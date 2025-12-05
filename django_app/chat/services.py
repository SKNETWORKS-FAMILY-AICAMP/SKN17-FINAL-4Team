# chat/services.py

from pathlib import Path
from typing import Any, Dict, Optional

import requests
from django.conf import settings


# settings.py 에서 MODEL_SERVER_URL 을 지정해두면 그걸 쓰고,
# 없으면 기본값으로 로컬 8000번 포트 사용
MODEL_SERVER_URL = getattr(settings, "MODEL_SERVER_URL", "http://127.0.0.1:8000")

# 타임아웃 기본값 (초)
TEXT_TIMEOUT = getattr(settings, "MODEL_SERVER_TEXT_TIMEOUT", 120)
IMAGE_TIMEOUT = getattr(settings, "MODEL_SERVER_IMAGE_TIMEOUT", 300)
RESET_TIMEOUT = getattr(settings, "MODEL_SERVER_RESET_TIMEOUT", 30)


def _build_url(path: str) -> str:
    base = MODEL_SERVER_URL.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def call_model_server_text(
    session_id: Optional[int],
    user_text: str,
    state_payload: Optional[Dict[str, Any]] = None,  # 현재는 사용 안 하지만 시그니처 유지
    request_more: bool = False,  # 현재는 사용 안 하지만 시그니처 유지
) -> Dict[str, Any]:
    """
    Django -> FastAPI 텍스트 대화 호출 래퍼.

    FastAPI /chat/text 스펙 (model_server.py 기준):

        요청 JSON:
        {
          "session_id": "optional string",
          "message": "사용자 입력"
        }

        응답 JSON:
        {
          "session_id": "...",
          "reply": "LLM 응답",
          "mode": "SMALLTALK|SURVEY|RECOMMEND",
          "llm_latency": 1.23,
          "debug_state_summary": "..."
        }

    chat/views.py 에서는 아래 키를 기대하므로
    여기서 포맷을 맞춰서 리턴한다:

        {
          "assistant_text": "...",
          "recommended_products": [...],
          "updated_session_state": {...}
        }
    """
    url = _build_url("/chat/text")

    payload = {
        "session_id": str(session_id) if session_id is not None else None,
        "message": user_text,
    }

    resp = requests.post(url, json=payload, timeout=TEXT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    assistant_text = data.get("reply", "")

    # 현재 model_server 응답에는 상품/상태가 구조화되어 있지 않으므로
    # 일단 빈 값으로 내려준다. (나중에 필요하면 확장)
    recommended_products: list = []
    updated_session_state: dict = {}

    return {
        "assistant_text": assistant_text,
        "recommended_products": recommended_products,
        "updated_session_state": updated_session_state,
        "_raw": data,  # 디버깅용
    }


def call_model_server_image(
    session_id: Optional[int],
    image_path: str,
    state_payload: Optional[Dict[str, Any]] = None,  # 현재는 사용 안 하지만 시그니처 유지
    is_want: bool = False,
) -> Dict[str, Any]:
    """
    Django -> FastAPI 이미지(VLM) 호출 래퍼.

    FastAPI /chat/image 스펙 (model_server.py 기준):

        form-data:
          - session_id (옵션)
          - is_want (bool)
          - file (이미지)

        응답 JSON:
        {
          "session_id": "...",
          "message": "VLM 결과 텍스트",
          "debug_state_summary": "..."
        }
    """
    url = _build_url("/chat/image")

    img_path = Path(image_path)
    if not img_path.is_file():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    data = {
        "session_id": str(session_id) if session_id is not None else "",
        "is_want": "true" if is_want else "false",
    }

    with img_path.open("rb") as f:
        files = {"file": (img_path.name, f)}
        resp = requests.post(url, data=data, files=files, timeout=IMAGE_TIMEOUT)

    resp.raise_for_status()
    data = resp.json()

    assistant_text = data.get("message", "")

    return {
        "assistant_text": assistant_text,
        "recommended_products": [],
        "updated_session_state": {},
        "_raw": data,
    }


def call_model_server_reset(session_id: int) -> Dict[str, Any]:
    """
    세션 전체 리셋 (/session/reset) 래퍼.

    FastAPI /session/reset 스펙:

        요청 JSON:
        { "session_id": "..." }

        응답 JSON:
        { "session_id": "...", "status": "reset" }
    """
    url = _build_url("/session/reset")
    payload = {"session_id": str(session_id)}

    resp = requests.post(url, json=payload, timeout=RESET_TIMEOUT)
    resp.raise_for_status()
    return resp.json()
