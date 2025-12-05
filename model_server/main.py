# main.py
"""
Qwen2.5-14B-Korean + Chroma RAG 기반
감성 기반 상품 추천 CLI 챗봇 (간단 상태머신 기반)

Flow:
1) 사용자 입력 받기
2) parse_user_query()로 카테고리/무드/예산/공간 추출 (턴 단위)
3) 모드 결정:
   - SMALLTALK : 잡담 모드
   - SURVEY    : 취향/공간/예산 질문 모드
   - RECOMMEND : RAG + 필터링 + 상품 추천 모드
4) SMALLTALK이 아닐 때만 세션 상태(session_state)에 누적 반영
5) 각 모드별로 LLM 호출
6) ::summary, ::image 같은 특수 명령 처리

설계 포인트:
- 이미지(VLM)로부터 얻는 무드/스타일 = 현재 상태(current)
- 사용자 텍스트로부터 얻는 무드 = 목표 상태(target)
- ::image -want 이미지_경로 → '원하는 분위기/레퍼런스' 이미지 기반 목표 상태(target_image_*)로 저장
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from config import (
    RAG_TOP_K,
    RECOMMEND_TOP_N,
)
from rag_retriever import RAGRetriever
from product_filter import filter_and_rank
from llm_core import chat, parse_user_query, DEFAULT_SYSTEM_PROMPT
from mood_vocab import snap_moods_to_vocab
from input_vlm import analyze_room_image  # VLM 모듈


# =========================
# 0. 유틸 함수들
# =========================

def _keep_korean(text: str) -> str:
    """문자열에서 한글과 공백만 남기고 나머지는 제거."""
    return re.sub(r"[^가-힣\s]", "", str(text)).strip()


def _clean_korean_list(values: List[str]) -> List[str]:
    """문자열 리스트에서 한글/공백만 남기고 비어 있거나 중복된 항목 제거."""
    out: List[str] = []
    for v in values:
        s = _keep_korean(v)
        if s and s not in out:
            out.append(s)
    return out


def _normalize_str_list(val: Any) -> List[str]:
    """
    문자열 / 리스트 / 튜플 등을 ['값1', '값2', ...]로 정규화.
    - 언어는 가리지 않고 그대로 보존 (영어도 유지).
    """
    if isinstance(val, str):
        chunks = re.split(r"[,\n/]", val)
        return [c.strip() for c in chunks if c.strip()]
    elif isinstance(val, (list, tuple)):
        out: List[str] = []
        for x in val:
            s = str(x).strip()
            if s and s not in out:
                out.append(s)
        return out
    else:
        return []


def _format_price(price_raw) -> str:
    """정수/문자/콤마 섞인 price를 '123,000원' 형태로 포맷."""
    if price_raw is None:
        return "가격 정보 없음"
    try:
        price = int(price_raw)
    except Exception:
        try:
            price = int(str(price_raw).replace(",", "").strip())
        except Exception:
            return "가격 정보 없음"
    return f"{price:,}원"


# =========================
# 1. 대화 상태 정의
# =========================

class ChatMode(Enum):
    SMALLTALK = auto()
    SURVEY = auto()
    RECOMMEND = auto()


@dataclass
class ChatState:
    """
    세션 전역 상태 (사용자 취향, 공간, 예산, 무드 등)

    - current_*        : 이미지/VLM 기반으로 추출된 "현재 방 상태"
    - target_*         : 사용자 텍스트에서 추출된 "목표/원하는 상태"
    - target_image_*   : 사용자가 올린 레퍼런스/원하는 분위기 이미지를 기반으로 추출한 "목표 상태"
    """

    # 공통 정보
    category: Optional[str] = None
    space: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None

    # 🔹 목표 무드 (사용자 텍스트)
    target_moods: List[str] = field(default_factory=list)
    unknown_target_moods: List[str] = field(default_factory=list)

    # 🔹 현재 무드 (이미지/VLM)
    current_moods: List[str] = field(default_factory=list)
    unknown_current_moods: List[str] = field(default_factory=list)

    # 현재 상태의 스타일/색감/재질/조명 (주로 VLM에서 옴)
    style_keywords: List[str] = field(default_factory=list)
    color_keywords: List[str] = field(default_factory=list)
    material_keywords: List[str] = field(default_factory=list)
    lighting_keywords: List[str] = field(default_factory=list)

    # VLM이 한 줄로 요약한 현재 방 분위기
    vlm_description: Optional[str] = None

    # 🔹 이미지 기반 "목표 상태" (레퍼런스/원하는 방/제품 사진)
    target_image_moods: List[str] = field(default_factory=list)
    unknown_target_image_moods: List[str] = field(default_factory=list)
    target_image_style_keywords: List[str] = field(default_factory=list)
    target_image_color_keywords: List[str] = field(default_factory=list)
    target_image_material_keywords: List[str] = field(default_factory=list)
    target_image_lighting_keywords: List[str] = field(default_factory=list)
    target_image_description: Optional[str] = None

    # 디버그 / 내부용
    last_intent: Optional[str] = None
    last_user_message: Optional[str] = None
    last_recommended_ids: List[str] = field(default_factory=list)

    # 🔸 실제 추천에 사용할 "타겟 무드"
    @property
    def effective_target_moods(self) -> List[str]:
        """
        추천/검색에서 사용할 목표 무드 우선순위:

        1) 사용자가 텍스트로 명시한 target_moods
        2) 사용자가 올린 '원하는 분위기' 레퍼런스 이미지의 무드(target_image_moods)
        3) 아무것도 없으면 현재 방 분위기(current_moods)를 기본 취향으로 가정
        """
        if self.target_moods:
            return self.target_moods
        if self.target_image_moods:
            return self.target_image_moods
        return self.current_moods

    # -------------------------
    # 업데이트 헬퍼들
    # -------------------------

    def update_from_parsed(self, parsed: Dict[str, Any]) -> None:
        """
        parse_user_query 결과를 세션 상태에 누적.

        ⚠️ 여기서 들어오는 무드는 전부 "사용자가 원하는 목표 무드"로 해석한다.
        """
        if parsed.get("category"):
            self.category = parsed["category"]

        if parsed.get("space"):
            self.space = parsed["space"]

        # 값이 들어온 것만 덮어쓰기
        if parsed.get("price_min") is not None:
            self.price_min = parsed["price_min"]
        if parsed.get("price_max") is not None:
            self.price_max = parsed["price_max"]

        # 목표 무드 누적 (이미 canonical로 들어옴)
        new_targets = parsed.get("moods") or []
        for m in new_targets:
            s = str(m).strip()
            if s and s not in self.target_moods:
                self.target_moods.append(s)

        # 존재하지 않는 목표 무드 표현도 누적
        new_unknown_targets = parsed.get("unknown_moods") or []
        for um in new_unknown_targets:
            s = str(um).strip()
            if s and s not in self.unknown_target_moods:
                self.unknown_target_moods.append(s)

    def merge(self, other: "ChatState") -> None:
        """
        다른 ChatState(예: VLM 결과)를 현재 세션에 병합.

        - category/space/price 정보는 기존 값이 없을 때만 채움
        - current_moods / 스타일 계열은 "현재 상태"로 흡수
        - target_moods / target_image_* 는 여기서 건드리지 않는다 (각각 별도 경로에서만 업데이트)
        """
        # 공통 필드
        if other.category and not self.category:
            self.category = other.category
        if other.space and not self.space:
            self.space = other.space

        if other.price_min is not None and self.price_min is None:
            self.price_min = other.price_min
        if other.price_max is not None and self.price_max is None:
            self.price_max = other.price_max

        # VLM 설명
        if other.vlm_description and not self.vlm_description:
            self.vlm_description = other.vlm_description

        # 현재 상태 계열 필드들 병합
        for field_name in [
            "current_moods",
            "unknown_current_moods",
            "style_keywords",
            "color_keywords",
            "material_keywords",
            "lighting_keywords",
        ]:
            current = getattr(self, field_name)
            incoming = getattr(other, field_name)
            for v in incoming:
                if v not in current:
                    current.append(v)


# =========================
# 2. 세션 전역 객체 + 대화 히스토리
# =========================

retriever = RAGRetriever()
session_state = ChatState()

# (user, assistant) 튜플 리스트
chat_history: List[Tuple[str, str]] = []


# =========================
# 3. 모드 결정 로직
# =========================

def decide_mode(
    user_text: str,
    parsed: Dict[str, Any],
    state: ChatState,
) -> ChatMode:
    """
    간단한 규칙 기반 모드 결정.

    - 인사 위주 문장 → SMALLTALK
    - '추천/골라/어울릴/뭐가 좋을/바꾸고 싶어/만들고 싶어' 등 + (공간/무드/예산/카테고리) → RECOMMEND
    - (이미지로 current_moods 있음) + (텍스트로 target moods 파싱됨) → RECOMMEND
    - 공간 + (최소/최대 예산 둘 중 하나) 있으면 → RECOMMEND
    - 그 외 → SURVEY
    """
    low = user_text.strip().lower()

    # 0) 인사 위주의 문장 (추천 키워드 없을 때만)
    if any(k in low for k in ["안녕", "hello", "hi", "ㅎㅇ"]) and "추천" not in low:
        return ChatMode.SMALLTALK

    # ---------- 이번 턴에서 파싱된 값 ----------
    parsed_space = parsed.get("space")
    parsed_price_min = parsed.get("price_min")
    parsed_price_max = parsed.get("price_max")
    parsed_moods = parsed.get("moods") or []
    parsed_category = parsed.get("category")

    has_parsed_mood = bool(parsed_moods)

    # ---------- 기존 state + 이번 턴 parsed 합쳐서 본 값 ----------
    has_space = bool(state.space or parsed_space)
    has_any_budget = (
        parsed_price_min is not None
        or parsed_price_max is not None
        or state.price_min is not None
        or state.price_max is not None
    )

    has_current_mood = bool(state.current_moods)

    # 이 대화가 지금까지 모아 둔 "실제 취향 정보"가 있는지 (이전 턴 기준)
    has_ref_image_pref = bool(
        state.target_image_moods
        or state.target_image_style_keywords
        or state.target_image_description
    )

    # 현재 방 사진이 업로드되어 있는지 여부 (무드/설명/스타일 중 하나라도 있으면 True)
    has_current_image = bool(
        state.current_moods
        or state.vlm_description
        or state.style_keywords
        or state.color_keywords
        or state.material_keywords
        or state.lighting_keywords
    )

    state_has_any_pref = (
        bool(state.target_moods or state.target_image_moods or has_ref_image_pref)
        or state.price_min is not None
        or state.price_max is not None
        or bool(state.category)
        or bool(state.space)
    )

    # 이번 턴까지 합쳐서 "취향 정보"가 있는지
    has_any_pref = (
        has_parsed_mood
        or bool(state.target_moods or state.target_image_moods or has_ref_image_pref)
        or has_any_budget
        or bool(parsed_category or state.category)
    )

    # 목표 무드 후보 (기존 effective_target_moods가 있으면 그걸 우선)
    if state.target_moods or state.target_image_moods or state.current_moods:
        effective_moods = state.effective_target_moods
    else:
        effective_moods = parsed_moods

    has_mood = bool(effective_moods)

    # ---------- 키워드 감지 ----------
    change_intent_keywords = [
        "분위기 바꾸고",
        "분위기를 바꾸고",
        "분위기를 좀 바꾸고",
        "따뜻하게 만들고",
        "따뜻한 분위기로 만들고",
        "이 방을 좀 더",
        "기분을 바꾸고",
        "무드를 바꾸고",
        "따뜻한 무드로",
    ]
    has_change_intent = any(k in user_text for k in change_intent_keywords)

    recommend_keywords = ["추천", "골라", "어울릴", "뭐가 좋을", "뭐가 어울릴지"]
    has_recommend_word = any(k in user_text for k in recommend_keywords)

    # ---------- 0) VLM + 이번 턴 텍스트 무드 → 바로 추천 ----------
    if has_current_mood and has_parsed_mood:
        return ChatMode.RECOMMEND
    
    # 🔵 (추가) 사진은 이미 있고, 추천 키워드까지 있으면 → 바로 추천 모드
    if has_current_image and has_recommend_word:
        return ChatMode.RECOMMEND

    # ---------- 0-1) 완전 포괄적인 "분위기 바꾸고 싶어"이면서, 아직 아무 정보도 없는 경우 → SURVEY 강제 ----------
    #   예: "방 분위기 바꾸고 싶어" (이전 턴에서도 아무것도 안 모인 상태)
    if has_change_intent and not state_has_any_pref and not has_parsed_mood and not has_any_budget and not parsed_category:
        return ChatMode.SURVEY

    # ---------- 0-2) 레퍼런스 이미지 + 추천 키워드 → 바로 추천 모드 ----------
    if has_recommend_word and has_ref_image_pref:
        return ChatMode.RECOMMEND

    # ---------- 1) 명시적인 추천/선택 키워드 ----------
    #   최소한 (공간/무드/예산/카테고리) + 어느 정도 취향 정보가 있을 때만 추천 모드
    if has_recommend_word and (has_space or has_mood or has_any_budget or parsed_category) and has_any_pref:
        return ChatMode.RECOMMEND

    # ---------- 2) "분위기를 바꾸고/만들고 싶다" ----------
    #   → 공간/무드/예산/카테고리 중 하나 + 실제 취향정보가 있을 때만 추천
    if has_change_intent and (has_space or has_mood or has_any_budget or parsed_category) and has_any_pref:
        return ChatMode.RECOMMEND

    # ---------- 3) 정보가 웬만큼 모인 경우 (공간 + 예산) ----------
    if has_space and has_any_budget:
        return ChatMode.RECOMMEND

    # ---------- 그 외에는 SURVEY ----------
    return ChatMode.SURVEY


# =========================
# 4. 추천 프롬프트 생성 유틸
# =========================

def build_recommendation_prompt(
    state: ChatState,
    products: List[Dict[str, Any]],
    user_text: str,
) -> str:
    """
    LLM에게 넘겨줄 '상품 리스트 + 컨텍스트' 문자열을 만든다.

    - 상품은 RAG + 필터링된 상위 N개
    - 현재 방 상태(current: 이미지/VLM)와
      사용자가 원하는 목표 상태(target: 텍스트/레퍼런스 이미지)를 명확히 분리해서 요약
    - "현재 → 목표"로 어떻게 바꿀지 설명하도록 유도
    """
    lines: List[str] = []
    lines.append("다음은 RAG로 검색된 후보 상품 목록이다.")
    lines.append("각 항목의 브랜드, 이름, 가격, 링크를 반드시 그대로 활용해라.\n")

    for i, p in enumerate(products, 1):
        brand = (p.get("brand_name") or "").strip()
        name = (p.get("product_name") or "").strip()
        link = (p.get("link_url") or "").strip()
        price_str = _format_price(p.get("price"))

        if brand:
            title = f"{brand} / {name}" if name else brand
        else:
            title = name or "(이름 없음)"

        lines.append(
            f"{i}. {title} (Price: {price_str}, Link: {link})"
        )

    # ==============================
    # 현재 방/공간 상태 (이미지/VLM 기준)
    # ==============================
    lines.append("\n[업로드한 방 사진에서 분석한 현재 상태(VLM)]")

    if state.vlm_description:
        lines.append(f"- VLM 분석 요약: {state.vlm_description}")
    if state.space:
        lines.append(f"- 공간: {state.space}  (이미지에서 인식한 공간)")
    if state.current_moods:
        lines.append(f"- 현재 무드(VLM 기준): {', '.join(state.current_moods)}")
    if state.style_keywords:
        lines.append(f"- 스타일 키워드: {', '.join(state.style_keywords)}")
    if state.color_keywords:
        lines.append(f"- 색감 키워드: {', '.join(state.color_keywords)}")
    if state.lighting_keywords:
        lines.append(f"- 조명 키워드: {', '.join(state.lighting_keywords)}")

    if not (state.vlm_description or state.space or state.current_moods):
        lines.append("- (이미지 정보 없음: 사용자가 사진을 올리지 않았을 수 있음)")

    # ==============================
    # 사용자가 올린 레퍼런스 이미지(원하는 분위기)
    # ==============================
    if state.target_image_moods or state.target_image_description:
        lines.append("\n[사용자가 올린 레퍼런스 이미지(원하는 분위기)]")

        if state.target_image_description:
            lines.append(f"- 레퍼런스 이미지 요약: {state.target_image_description}")
        if state.target_image_moods:
            lines.append(
                f"- 레퍼런스 이미지 기준 목표 무드: {', '.join(state.target_image_moods)}"
            )
        if state.target_image_style_keywords:
            lines.append(
                f"- 스타일 키워드(레퍼런스): {', '.join(state.target_image_style_keywords)}"
            )
        if state.target_image_color_keywords:
            lines.append(
                f"- 색감 키워드(레퍼런스): {', '.join(state.target_image_color_keywords)}"
            )

    # ==============================
    # 사용자가 말한 목표 상태 (텍스트 기준)
    # ==============================
    lines.append("\n[사용자가 이번 턴에서 말한 목표 상태(텍스트)]")
    target_moods = state.target_moods
    effective_targets = state.effective_target_moods

    if target_moods:
        lines.append(f"- 목표 무드(텍스트): {', '.join(target_moods)}")
    elif effective_targets:
        # 사용자가 따로 말한 적은 없지만, 이미지 등에서 파생된 기본 취향으로 가정
        lines.append(
            f"- 목표 무드(기본 취향/이미지 기준): {', '.join(effective_targets)}"
        )
    else:
        lines.append("- 목표 무드: 아직 명시되지 않음")

    if state.price_min is not None or state.price_max is not None:
        pm = state.price_min
        px = state.price_max
        if pm is not None and px is not None:
            lines.append(f"- 예산: 약 {pm:,}원 ~ {px:,}원")
        elif pm is not None:
            lines.append(f"- 예산: 최소 {pm:,}원")
        elif px is not None:
            lines.append(f"- 예산: 최대 {px:,}원")
    if state.category:
        lines.append(f"- 희망 카테고리: {state.category}")

    # 사용자의 이번 턴 발화
    lines.append("\n[사용자의 이번 턴 요청 문장]")
    lines.append(user_text)

    # ==============================
    # 답변 스타일 지침 (자연어)
    # ==============================
    lines.append(
        "\n위 정보와 상품 후보들을 바탕으로 다음 지침을 지켜서 답변해라.\n"
        "반드시 아래와 같은 구조를 지켜라.\n\n"
        "[추천 요약]\n"
        "- 여기에서는 구체적인 브랜드명/상품명을 쓰지 말고,\n"
        "  '베이지 러그', '우드톤 트레이', '앰버톤 조명'처럼 카테고리/형용사 위주의\n"
        "  개념적인 아이템만 언급해라.\n\n"
        "[추천 상품]\n"
        "- 이 섹션에서만 구체적인 상품을 나열한다.\n"
        "- 위에서 제공한 후보 상품 목록의 줄을 그대로 활용해서,\n"
        "  1번, 2번, 3번 형태로 다시 정리해라.\n"
        "- 각 항목은 반드시 다음 형식을 따라야 한다:\n"
        "  \"1. 브랜드 / 상품명 (Price: 00,000원, Link: URL)\"\n"
        "- 이때 '브랜드/상품명/가격/링크'는 반드시 위에서 제공된 후보 목록에서 온 값을\n"
        "  한 글자도 바꾸지 말고 그대로 복사해 사용해야 한다.\n"
        "- 후보 목록에 없는 브랜드/상품명/가격/링크를 새로 만들지 마라.\n\n"
        "추가 규칙:\n"
        "1. 답변의 첫 문장은 반드시 사용자가 올린 이미지(현재 방 사진 또는 레퍼런스 사진)를 언급하면서 시작해라.\n"
        "   예: \"올려주신 레퍼런스 사진을 보면 따뜻한 베이지 톤의 침실이네요.\" 와 같이 시작.\n"
        "2. 이어서 1~2문장으로 현재 방/공간의 분위기를 요약해라.\n"
        "3. 그 다음, 사용자가 이번 턴에서 말한 목표 분위기(무드)를 짧게 정리해라.\n"
        "4. 그 뒤에, [추천 요약] 섹션에서 어떤 종류의 아이템(러그, 쿠션, 조명 등)을\n"
        "   어떻게 배치하면 현재 분위기에서 목표 분위기로 전환되는지 설명해라.\n"
        "5. [추천 상품] 섹션에서는 후보 상품 목록에서 최소 3개 이상을 골라,\n"
        "   각 상품의 브랜드/이름/가격/링크를 위에서 제공된 텍스트 그대로 사용해서 나열해라.\n"
        "6. 후보 목록에 조건에 맞는 상품이 충분하지 않다면, 있는 만큼만 정직하게 나열하고\n"
        "   '현재 데이터에 없는 상품입니다'라고 솔직하게 말해라. 절대로 상상으로 채우지 마라.\n"
        "7. 전체 톤은 인테리어 전문가가 친절하게 조언해 주는 느낌으로, 과장 없이 구체적으로 설명해라.\n"
        "8. 절대로 [제품 브랜드], [제품 이름], [제품 링크], [제품 가격] 같은 플레이스홀더나 "
        "가상의 값/임의의 브랜드, 이름, 가격, 링크를 만들지 마라. "
        "반드시 위에서 번호를 붙여 나열한 후보 상품 목록의 브랜드/이름/가격/링크만 그대로 사용해라."
    )

    return "\n".join(lines)



# =========================
# 5. 모드별 응답 생성
# =========================

def handle_smalltalk(user_text: str, history: List[Tuple[str, str]]) -> tuple[str, float]:
    system_prompt = (
        "너는 감성 기반 인테리어 상품 추천 챗봇인데, "
        "지금은 사용자의 잡담/인사에만 자연스럽게 응답하면 돼. "
        "너무 길지 않게, 2~3문장 정도로 답해줘."
    )
    t0 = time.time()
    answer = chat(
        history=history,
        user_input=user_text,
        system_prompt=system_prompt,
    )
    elapsed = time.time() - t0
    return answer, elapsed


def handle_survey(
    user_text: str,
    state: ChatState,
    history: List[Tuple[str, str]],
) -> tuple[str, float]:
    """
    아직 정보가 부족할 때:
    - 공간(거실/침실/책상 등)
    - 예산 범위
    - 원하는 분위기/색감/재질
    등을 자연스럽게 물어봐 주는 모드.
    """

    # 🔹 현재 방 사진 / 레퍼런스 이미지 업로드 여부 파악
    has_current_image = bool(
        state.current_moods
        or state.vlm_description
        or state.style_keywords
        or state.color_keywords
        or state.material_keywords
        or state.lighting_keywords
    )
    has_ref_image = bool(
        state.target_image_moods
        or state.target_image_description
        or state.target_image_style_keywords
        or state.target_image_color_keywords
        or state.target_image_material_keywords
        or state.target_image_lighting_keywords
    )

    missing_fields: List[str] = []
    if not state.space:
        missing_fields.append("공간 정보(거실/침실/작업실 등)")
    if state.price_min is None and state.price_max is None:
        missing_fields.append("예산 범위")
    # 목표 무드가 아직 없을 때만 물어봄 (현재 무드는 이미지로 알 수 있으니)
    if not state.target_moods and not state.target_image_moods:
        missing_fields.append("원하는 목표 무드/분위기")
    if not state.color_keywords:
        missing_fields.append("선호하는 색감/색상")
    if not state.material_keywords:
        missing_fields.append("선호하는 재질(원목, 패브릭 등)")

    lines = [
        "너는 감성 기반 인테리어 상품 추천 챗봇이야.",
        "",
        "아래 정보는 시스템이 이미 파악한 상태야. 이 정보를 최대한 활용해서, "
        "사용자에게 부족한 정보만 자연스럽게 질문해 줘.",
        "",
        f"- 현재 방 사진 업로드 여부: {'있음' if has_current_image else '없음'}",
        f"- 레퍼런스(원하는 분위기) 이미지 업로드 여부: {'있음' if has_ref_image else '없음'}",
        "",
        "중요 규칙:",
        "1) 사용자가 이미 방 사진이나 레퍼런스 이미지를 업로드한 상태라면, "
        "   절대로 '사진을 다시 보내 달라'거나 '사진을 공유해 달라'고 말하지 마.",
        "2) 대신 아래에 정리된 VLM 기반 정보(current_*, target_image_*)를 활용해서 "
        "   사용자의 공간과 취향을 이해했다고 가정하고 질문을 이어가.",
        "3) 사용자가 아직 사진을 전혀 올리지 않았고, 명시적으로 '사진 보내줄까?'라고 물을 때만 "
        "   부드럽게 사진 업로드를 제안해도 된다.",
        "",
        "현재까지 알고 있는 사용자 정보는 다음과 같아.",
        f"- 공간: {state.space or '아직 정보 없음'}",
        f"- 현재 무드(VLM): {', '.join(state.current_moods) if state.current_moods else '아직 정보 없음'}",
        f"- 목표 무드(텍스트): {', '.join(state.target_moods) if state.target_moods else '아직 정보 없음'}",
        f"- 목표 무드(이미지): {', '.join(state.target_image_moods) if state.target_image_moods else '아직 정보 없음'}",
        f"- 예산: {state.price_min or '미정'} ~ {state.price_max or '미정'}",
        f"- 색감: {', '.join(state.color_keywords) if state.color_keywords else '아직 정보 없음'}",
        f"- 재질: {', '.join(state.material_keywords) if state.material_keywords else '아직 정보 없음'}",
        "",
    ]

    if missing_fields:
        lines.append(
            "특히 아직 모르는 정보는 다음과 같아: "
            + ", ".join(missing_fields)
        )
        lines.append(
            "이미 값이 채워진 항목(공간, 예산, 목표 무드, 색감, 재질 등)은 절대로 다시 묻지 말고, "
            "위에 나열된 '아직 정보 없음' 항목 중에서 1~2가지만 자연스럽게 질문해 줘."
        )
    else:
        lines.append(
            "이미 정보가 꽤 모였으니, 추가로 있으면 좋을만한 정보 한 가지만 가볍게 확인해 줘. "
            "질문은 1개만, 짧은 한국어 대화체로."
        )

    lines.append(
        "한 번에 너무 많은 걸 묻지 말고, 자연스러운 한국어 대화체로 짧게 질문해 줘."
    )

    system_prompt = "\n".join(lines)

    t0 = time.time()
    answer = chat(
        history=history,
        user_input=user_text,
        system_prompt=system_prompt,
    )
    elapsed = time.time() - t0
    return answer, elapsed



def handle_recommend(
    user_text: str,
    state: ChatState,
    history: List[Tuple[str, str]],


) -> tuple[str, float]:
    """
    RAG + 상품 필터링/랭킹을 통해 실제 추천을 생성하는 모드.
    """

    # 🔒 안전장치: 추천에 쓸 만한 정보가 하나도 없으면 SURVEY로 돌려보내기
    if (
        not state.effective_target_moods
        and state.price_min is None
        and state.price_max is None
        and not state.category
        and not state.space
    ):
        # 최소한의 취향/공간 정보가 없으니, 우선 질문 모드로 한 번 더 유도
        return handle_survey(user_text, state, history)

    # 0) 상반되거나 사전에 없는 "목표 무드"에 대한 안내 메시지 구성
    explain_prefix = ""
    target_moods = state.target_moods
    unknown_targets = state.unknown_target_moods

    if target_moods and unknown_targets:
        known_str = ", ".join(target_moods)
        unknown_str = ", ".join(unknown_targets)
        explain_prefix = (
            f"요청해 주신 목표 무드({known_str} / {unknown_str})를 모두 동시에 만족하는 상품을 찾기는 어려워서,\n"
            f"현재 시스템이 인식할 수 있는 무드인 '{known_str}' 분위기를 중심으로 추천을 드릴게요.\n\n"
        )
    elif unknown_targets:
        unknown_str = ", ".join(unknown_targets)
        explain_prefix = (
            f"요청해 주신 목표 무드 '{unknown_str}'는(은) 현재 시스템의 무드 사전에 없어서,\n"
            "가장 비슷한 분위기의 상품 위주로 추천을 드릴게요.\n\n"
        )

    # 1) RAG 검색 쿼리 구성
    query = user_text

    effective_targets = state.effective_target_moods
    if effective_targets:
        query += "\n\n[사용자가 원하는 목표 무드] " + ", ".join(effective_targets)
    if state.current_moods:
        query += "\n[현재 방 무드(VLM)] " + ", ".join(state.current_moods)
    if state.space:
        query += f"\n[공간] {state.space}"
    if state.category:
        query += f"\n[희망 카테고리] {state.category}"
    # 레퍼런스 이미지 기반 정보도 검색 쿼리에 반영
    if state.target_image_style_keywords:
        query += "\n[레퍼런스 이미지 스타일 키워드] " + ", ".join(state.target_image_style_keywords)
    if state.target_image_description:
        query += "\n[레퍼런스 이미지 요약] " + state.target_image_description

    # 🔹 retriever.search: rag_retriever.py에서 state 기반 필터까지 걸어줄 수 있음
    retrieved = retriever.search(query, state=state)
    print(f"[DEBUG] RAG retrieved: {len(retrieved)}개")
    if retrieved:
        print("[DEBUG] sample retrieved[0]:", retrieved[0])

    # 2) 상품 필터링/랭킹
    ranked_all = filter_and_rank(
        products=retrieved,
        state=state,
    )
    print(f"[DEBUG] ranked_all: {len(ranked_all)}개")
    if ranked_all:
        print("[DEBUG] sample ranked_all[0]:", ranked_all[0])

    # 예산 정보가 있는데 랭킹 결과가 0개인 경우 → 예산대에 맞는 상품 없음
    has_budget = state.price_min is not None or state.price_max is not None
    if has_budget and not ranked_all:
        pm = state.price_min
        px = state.price_max
        if pm is not None and px is not None:
            budget_str = f"{pm:,}원 ~ {px:,}원"
        elif pm is not None:
            budget_str = f"{pm:,}원 이상"
        elif px is not None:
            budget_str = f"{px:,}원 이하"
        else:
            budget_str = "요청 예산"

        msg = (
            f"지금 말씀하신 예산대({budget_str})에 딱 맞는 상품은 "
            "현재 데이터셋에서 찾지 못했어요.\n"
            "데이터에 등록된 상품 가격대가 그 범위와 많이 다를 가능성이 커요.\n\n"
            "예산 범위를 조금 넓게 말해 주거나, 다른 카테고리/무드로 다시 요청해 줄래요?"
        )
        if explain_prefix:
            msg = explain_prefix + msg
        return msg, 0.0

    # 예산 조건이 없고, 그냥 아무 것도 못 찾은 경우의 일반 메시지
    if not ranked_all:
        msg = (
            "지금 조건에 딱 맞는 상품을 찾기 어렵네요.\n"
            "예산 범위나 원하는 무드를 조금 더 넓게 알려주면, "
            "더 다양한 상품을 추천해 볼게요!"
        )
        if explain_prefix:
            msg = explain_prefix + msg
        return msg, 0.0

    # 상위 N개만 LLM에 넘김
    ranked = ranked_all[:RECOMMEND_TOP_N]

    # 이번 턴에 추천한 상품 id 저장 (다음 턴에 중복 페널티)
    state.last_recommended_ids = [
        p.get("product_id")
        for p in ranked
        if p.get("product_id")
    ]

    # 3) LLM으로 자연스러운 설명 생성
    recommendation_prompt = build_recommendation_prompt(state, ranked, user_text)

    t0 = time.time()
    answer = chat(
        history=history,
        user_input=recommendation_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        do_sample=False,  # 🔴 추천에서는 샘플링 끄고 greedy로 고정
    )
    elapsed = time.time() - t0

    # 안내 프리픽스 붙이기
    if explain_prefix:
        answer = explain_prefix + answer

    return answer, elapsed


# =========================
# 6. 특수 명령 처리 (::summary, ::image ... )
# =========================

def render_summary(state: ChatState) -> str:
    lines = [
        "[디버그 요약]",
        "",
        "※ current_*  = 이미지/VLM에서 추출한 '현재 방 상태'",
        "※ target_*   = 사용자가 텍스트로 말한 '목표/원하는 상태'",
        "※ target_image_* = 사용자가 올린 레퍼런스/원하는 분위기 이미지 기반 '목표 상태'",
        "※ effective_target_moods = 실제 추천에 사용하는 최종 목표 무드",
        "",
        "[공통 정보]",
        f"- category : {state.category}",
        f"- space    : {state.space}",
        f"- price    : {state.price_min} ~ {state.price_max}",
        "",
        "[현재 상태 (이미지/VLM 기반)]",
        f"- current_moods           : {', '.join(state.current_moods) if state.current_moods else '없음'}",
        f"- unknown_current_moods   : {', '.join(state.unknown_current_moods) if state.unknown_current_moods else '없음'}",
        f"- style_keywords          : {', '.join(state.style_keywords) if state.style_keywords else '없음'}",
        f"- color_keywords          : {', '.join(state.color_keywords) if state.color_keywords else '없음'}",
        f"- material_keywords       : {', '.join(state.material_keywords) if state.material_keywords else '없음'}",
        f"- lighting_keywords       : {', '.join(state.lighting_keywords) if state.lighting_keywords else '없음'}",
        f"- vlm_description         : {state.vlm_description or '없음'}",
        "",
        "[목표 상태 (이미지/VLM 기반)]",
        f"- target_image_moods        : {', '.join(state.target_image_moods) if state.target_image_moods else '없음'}",
        f"- unknown_target_image_moods: {', '.join(state.unknown_target_image_moods) if state.unknown_target_image_moods else '없음'}",
        f"- target_image_style        : {', '.join(state.target_image_style_keywords) if state.target_image_style_keywords else '없음'}",
        f"- target_image_color        : {', '.join(state.target_image_color_keywords) if state.target_image_color_keywords else '없음'}",
        f"- target_image_material     : {', '.join(state.target_image_material_keywords) if state.target_image_material_keywords else '없음'}",
        f"- target_image_lighting     : {', '.join(state.target_image_lighting_keywords) if state.target_image_lighting_keywords else '없음'}",
        f"- target_image_description  : {state.target_image_description or '없음'}",
        "",
        "[목표 상태 (사용자 텍스트 기반)]",
        f"- target_moods            : {', '.join(state.target_moods) if state.target_moods else '없음'}",
        f"- unknown_target_moods    : {', '.join(state.unknown_target_moods) if state.unknown_target_moods else '없음'}",
        f"- effective_target_moods  : {', '.join(state.effective_target_moods) if state.effective_target_moods else '없음'}",
        "",
        f"- last_intent             : {state.last_intent}",
    ]
    return "\n".join(lines)


def handle_image_command(arg: str, state: ChatState) -> str:
    """
    ::image 이미지_경로         → 현재 방 사진 (current_* 업데이트)
    ::image -want 이미지_경로   → 원하는 분위기/제품/방 사진 (target_image_* 업데이트)

    예:
      ::image "C:\\my_python\\Final_Project\\room.jpg"
      ::image -want "C:\\my_python\\Final_Project\\ref_cushion.jpg"
    """
    arg = arg.strip()

    # 1) -want 옵션 파싱
    is_want_image = False
    raw_part = arg

    if raw_part.startswith("-want"):
        is_want_image = True
        raw_part = raw_part[len("-want"):].strip()

    raw_path = raw_part.strip().strip('"').strip("'")
    image_path = Path(raw_path)

    if not image_path.is_file():
        return f"[VLM] 이미지 파일을 찾을 수 없어요: {image_path}"

    mode_str = "원하는 레퍼런스/제품/방 사진" if is_want_image else "현재 방 사진"
    print(f"[VLM] {mode_str} 무드 분석 중... ({image_path})")

    # VLM 실행 시간 측정
    t0 = time.time()
    info = analyze_room_image(str(image_path))
    t1 = time.time()
    elapsed = t1 - t0

    # 🔍 디버그 출력
    print("[DEBUG] raw VLM result from analyze_room_image():")
    print(info)

    # ==============================
    # 0) 인테리어/소품 이미지인지, 유효 결과인지 먼저 검사
    # ==============================
    image_type = (info.get("image_type") or "").strip().lower()
    has_any_feature = any([
        info.get("space_ko"),
        info.get("space_en"),
        info.get("style_keywords"),
        info.get("color_keywords"),
        info.get("mood_keywords"),
        info.get("material_keywords"),
        info.get("lighting_keywords"),
    ])

    # 인테리어가 아닌 사진이거나, 모델이 아무 정보도 못 채운 경우
    is_non_interior = bool(image_type and image_type != "interior_room")
    is_empty_result = (not image_type) and (not has_any_feature)

    if is_non_interior or is_empty_result:
        # VLM이 이미 이유를 남겼으면 그대로 사용, 없으면 기본 문장 사용
        reason = info.get("overall_comment_ko") or (
            "이 이미지는 방 전체나 인테리어 소품을 보여주는 사진이 "
            "아니라고 판단해서, 무드 키워드를 추출하지 않았고 "
            "추천 상태에도 반영하지 않았어요."
        )

        if is_want_image:
            return (
                "[VLM] 이 이미지는 인테리어 레퍼런스용으로 사용하기 어렵다고 판단했어요.\n"
                f"- 사유: {reason}\n"
                "  → 이 이미지는 목표 무드(레퍼런스) 정보에 포함하지 않았습니다.\n"
                f"- VLM 분석 소요 시간: {elapsed:.1f}초"
            )
        else:
            return (
                "[VLM] 이 이미지는 현재 방 분위기를 파악하기 어려워요.\n"
                f"- 사유: {reason}\n"
                "  → 이 이미지는 현재 방 무드/스타일 정보에 반영하지 않았습니다.\n"
                f"- VLM 분석 소요 시간: {elapsed:.1f}초"
            )

    # input_vlm.analyze_room_image() 반환 스키마:
    # {
    #   "image_type": ...,
    #   "space_ko": ...,
    #   "space_en": ...,
    #   "style_keywords": [...],
    #   "color_keywords": [...],
    #   "mood_keywords": [...],
    #   "material_keywords": [...],
    #   "lighting_keywords": [...],
    #   "overall_comment_ko": ...
    # }
    vlm_space = info.get("space_ko") or info.get("space_en") or None

    # 🔹 원본 키워드 그대로 정규화 (언어 상관없이 보존)
    raw_moods = _normalize_str_list(info.get("mood_keywords", []) or [])
    vlm_style = _normalize_str_list(info.get("style_keywords", []) or [])
    vlm_color = _normalize_str_list(info.get("color_keywords", []) or [])
    vlm_material = _normalize_str_list(info.get("material_keywords", []) or [])
    vlm_lighting = _normalize_str_list(info.get("lighting_keywords", []) or [])

    # space 는 여전히 한글만 남기기
    if vlm_space:
        vlm_space = _keep_korean(vlm_space)

    # 🔹 무드만은 "한글 버전"을 따로 만들어서 사전에 스냅
    moods_ko = _clean_korean_list(raw_moods)
    canonical_moods, unknown_moods_ko = snap_moods_to_vocab(moods_ko)

    # 🔹 사전에도 없고, 한글로도 못 만든(영어/기타) 무드들까지 unknown에 포함
    non_ko_moods = [m for m in raw_moods if _keep_korean(m) == ""]
    unknown_moods_extra = [m for m in non_ko_moods if m not in unknown_moods_ko]
    unknown_moods = list(dict.fromkeys(unknown_moods_ko + unknown_moods_extra))

    # 🔹 실제 state에 반영할 무드:
    #   1순위: canonical_moods (정제된 한국어 무드)
    #   2순위: moods_ko (한글 버전)
    #   3순위: raw_moods (영어/기타라도 그냥 사용)
    if canonical_moods:
        effective_moods = canonical_moods
    elif moods_ko:
        effective_moods = moods_ko
    else:
        effective_moods = raw_moods

    # VLM가 생성한 설명 텍스트
    vlm_description = (
        info.get("overall_comment_ko")
        or info.get("description")
        or None
    )

    # ==============================
    # 2) 현재 방 사진 모드 (기존 동작)
    # ==============================
    if not is_want_image:
        vlm_state = ChatState(
            current_moods=effective_moods,
            unknown_current_moods=unknown_moods,
            space=vlm_space,
            style_keywords=vlm_style,
            color_keywords=vlm_color,
            material_keywords=vlm_material,
            lighting_keywords=vlm_lighting,
            vlm_description=vlm_description,
        )
        state.merge(vlm_state)

        lines = ["[VLM] 방 이미지(현재 상태) 분석 결과를 반영했어요."]
        if vlm_space:
            lines.append(f"- 공간 인식: {vlm_space}")
        if raw_moods:
            lines.append(f"- VLM이 감지한 원본 무드 후보: {', '.join(raw_moods)}")
        if canonical_moods:
            lines.append(f"- 정제된 무드 사전에 맞춰 정규화한 현재 무드: {', '.join(canonical_moods)}")
        if unknown_moods:
            lines.append(
                "- 사전에 없는 무드도 일부 있어서, 비슷한 분위기의 상품으로 최대한 근접하게 추천해 볼게요."
            )
        if vlm_description:
            lines.append(f"- 이미지 설명(VLM): {vlm_description}")

        lines.append(f"- VLM 분석 소요 시간: {elapsed:.1f}초")
        return "\n".join(lines)

    # ==============================
    # 3) -want 모드: '목표 상태(이미지 기반)'로 반영
    # ==============================
    # 공간은 아직 정해져 있지 않다면, 레퍼런스 이미지의 공간을 기본값으로 사용
    if vlm_space and not state.space:
        state.space = vlm_space

    state.target_image_moods = effective_moods
    state.unknown_target_image_moods = unknown_moods
    state.target_image_style_keywords = vlm_style
    state.target_image_color_keywords = vlm_color
    state.target_image_material_keywords = vlm_material
    state.target_image_lighting_keywords = vlm_lighting
    state.target_image_description = vlm_description

    lines = ["[VLM] '원하는 분위기' 레퍼런스 이미지 분석 결과를 반영했어요."]
    if vlm_space:
        lines.append(f"- 레퍼런스 이미지 공간 인식: {vlm_space}")
    if raw_moods:
        lines.append(f"- VLM이 감지한 원본 목표 무드 후보: {', '.join(raw_moods)}")
    if canonical_moods:
        lines.append(f"- 정제된 무드 사전에 맞춰 정규화한 목표 무드: {', '.join(canonical_moods)}")
    if unknown_moods:
        lines.append(
            "- 사전에 없는 무드 표현도 있어서, 비슷한 분위기의 상품으로 최대한 근접하게 추천해 볼게요."
        )
    if vlm_description:
        lines.append(f"- 레퍼런스 이미지 설명(VLM): {vlm_description}")

    lines.append(f"- VLM 분석 소요 시간: {elapsed:.1f}초")

    return "\n".join(lines)



# =========================
# 7. 메인 루프
# =========================

def main() -> None:
    print("===============================================")
    print("  감성 기반 상품 추천 챗봇 (RAG + Qwen2.5-14B-Korean)")
    print("   - 종료하려면 'exit' 또는 'quit' 입력")
    print("   - 디버그 요약: '::summary' 입력")
    print("   - 방 이미지 무드 추출: '::image 이미지_경로' 입력 (현재 방 사진)")
    print("   - 레퍼런스/원하는 분위기 이미지: '::image -want 이미지_경로' 입력")
    print("   - 상태 초기화: '::reset_all' 또는 '::reset_moods' 입력")
    print("   - 예: '방 분위기를 바꾸고 싶어'라고 말하면, 내가 먼저 공간/예산/무드를 물어볼 거야.")
    print("===============================================")

    global session_state, chat_history

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[시스템] 종료합니다.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("[시스템] 안녕히 가세요!")
            break

        # 특수 명령 처리
        if user_text.startswith("::summary"):
            print(render_summary(session_state))
            continue

        if user_text.startswith("::image"):
            arg = user_text[len("::image"):].strip()
            resp = handle_image_command(arg, session_state)
            print(resp)
            continue

        # 세션 전체 리셋
        if user_text.startswith("::reset_all"):
            session_state = ChatState()
            chat_history = []
            print("[시스템] 세션 상태와 대화 히스토리를 모두 초기화했어요.")
            continue

        # 무드/스타일 관련 키워드 + VLM 설명만 리셋
        if user_text.startswith("::reset_moods"):
            # 텍스트 기반 목표 무드
            session_state.target_moods.clear()
            session_state.unknown_target_moods.clear()

            # 현재 방 상태 무드/스타일
            session_state.current_moods.clear()
            session_state.unknown_current_moods.clear()
            session_state.style_keywords.clear()
            session_state.color_keywords.clear()
            session_state.material_keywords.clear()
            session_state.lighting_keywords.clear()
            session_state.vlm_description = None

            # 이미지 기반 목표 상태
            session_state.target_image_moods.clear()
            session_state.unknown_target_image_moods.clear()
            session_state.target_image_style_keywords.clear()
            session_state.target_image_color_keywords.clear()
            session_state.target_image_material_keywords.clear()
            session_state.target_image_lighting_keywords.clear()
            session_state.target_image_description = None

            print("[시스템] 현재/목표 무드와 스타일 관련 키워드(VLM/텍스트/레퍼런스 이미지)를 초기화했어요.")
            continue

        # 일반 대화 처리
        session_state.last_user_message = user_text

        # 1) 이번 턴 파싱
        parsed = parse_user_query(user_text)

        # 2) 파싱 결과와 기존 state를 기반으로 모드 결정
        mode = decide_mode(user_text, parsed, session_state)
        session_state.last_intent = mode.name

        # 3) SMALLTALK이 아닐 때만 state에 누적
        if mode != ChatMode.SMALLTALK:
            session_state.update_from_parsed(parsed)

        # 4) 모드별 응답 생성
        if mode == ChatMode.SMALLTALK:
            answer, llm_sec = handle_smalltalk(user_text, chat_history)
        elif mode == ChatMode.SURVEY:
            answer, llm_sec = handle_survey(user_text, session_state, chat_history)
        else:
            answer, llm_sec = handle_recommend(user_text, session_state, chat_history)

        print(f"\nBot: {answer}\n")
        print(f"[LLM] 응답 생성 소요 시간: {llm_sec:.1f}초\n")

        # 히스토리 업데이트 (다음 턴 컨텍스트용)
        chat_history.append((user_text, answer))


if __name__ == "__main__":
    main()
