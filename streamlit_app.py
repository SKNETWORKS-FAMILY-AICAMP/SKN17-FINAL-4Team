# streamlit_app.py
"""
RAG 기반 대화형 상품 추천 데모 (Streamlit 버전)

- 로컬에 다운로드 해 둔 Qwen2.5-14B-Korean (8bit, llm_core.py에서 로딩)
- Chroma Vector DB + SentenceTransformer RAG (rag_retriever.py)
- 상품 필터링 / 랭킹 (product_filter.py)
- CLI main.py의 상태머신 로직을 Streamlit용으로 옮긴 버전

실행 방법 (Final_Project 폴더에서):
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st

from config import (
    RAG_TOP_K,
    RECOMMEND_TOP_N,
)
from rag_retriever import RAGRetriever
from product_filter import filter_and_rank
from llm_core import chat, parse_user_query


# =========================
# 0. Streamlit 기본 설정
# =========================

st.set_page_config(
    page_title="감성 기반 인테리어 추천 데모",
    page_icon="🛋️",
    layout="wide",
)


# =========================
# 1. 상태 정의 (CLI와 동일)
# =========================

class ChatMode(Enum):
    SMALLTALK = auto()   # 잡담 모드
    SURVEY = auto()      # 취향/공간/예산 질문 모드
    RECOMMEND = auto()   # 상품 추천 모드


@dataclass
class ChatState:
    # 사용자가 지금까지 제공한 선호 정보(세션 누적)
    category: Optional[str] = None
    moods: List[str] = field(default_factory=list)
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    space: Optional[str] = None  # 꾸미고 싶은 공간(책상 근처, 침실, 거실 등)

    # 상태머신 관련
    mode: ChatMode = ChatMode.SMALLTALK

    # 반복 질문 완화용: 마지막 설문 상태 시그니처
    last_survey_signature: Optional[str] = None


# =========================
# 2. 유틸 함수들 (CLI main.py와 동일/유사)
# =========================

def format_won(price: Any) -> str:
    try:
        p = int(price)
    except Exception:
        return str(price)
    return f"{p:,}원"


def build_retrieval_query(user_text: str, state: ChatState) -> str:
    """
    RAG 쿼리로 넘길 한 줄짜리 텍스트 생성
    (유저 원문 + 파싱된 정보들을 섞어서 좀 더 힌트 제공)
    """
    parts = [f"사용자 요청: {user_text}"]

    if state.category:
        parts.append(f"원하는 카테고리: {state.category}")
    if state.moods:
        parts.append(f"원하는 무드: {', '.join(state.moods)}")
    if state.price_min is not None or state.price_max is not None:
        parts.append(
            f"예산 범위: {state.price_min if state.price_min is not None else '미상'}"
            f" ~ {state.price_max if state.price_max is not None else '미상'} 원"
        )
    if state.space:
        parts.append(f"꾸미고 싶은 공간: {state.space}")

    return " | ".join(parts)


def state_from_parsed(parsed: Dict[str, Any]) -> ChatState:
    """
    parse_user_query 결과를 한 턴짜리 임시 상태(ChatState)로 변환
    (세션 누적이 아니라 '이번 턴에서 새로 얻은 정보'만 담긴 상태)
    """
    return ChatState(
        category=parsed.get("category"),
        moods=parsed.get("moods") or [],
        price_min=parsed.get("price_min"),
        price_max=parsed.get("price_max"),
        space=parsed.get("space"),
    )


def update_session_state(session_state: ChatState, turn_state: ChatState) -> None:
    """
    세션 누적 상태(session_state)에 이번 턴 상태(turn_state) 반영
    - None / 빈 값은 덮어쓰지 않고
    - 새로 들어온 정보만 채워 넣음
    """
    if turn_state.category:
        session_state.category = turn_state.category

    if turn_state.moods:
        existing = set(session_state.moods)
        for m in turn_state.moods:
            if m not in existing:
                session_state.moods.append(m)

    if turn_state.price_min is not None:
        session_state.price_min = turn_state.price_min
    if turn_state.price_max is not None:
        session_state.price_max = turn_state.price_max

    if turn_state.space:
        session_state.space = turn_state.space


def infer_category_from_space_rag(retriever: RAGRetriever, session_state: ChatState) -> None:
    """
    RAG(Chroma + SentenceTransformer)를 이용해서
    space(책상 근처, 침실, 거실 등) → category_id를 유추한다.

    - session_state.category 가 이미 있으면 건드리지 않음
    - session_state.space 가 없으면 아무 것도 안 함
    - 상위 RAG 결과의 category_id 를 sim_score 가중합으로 집계해서
      가장 점수가 높은 category_id 하나를 세팅
    """
    if session_state.category or not session_state.space:
        return

    query_text = (
        f"{session_state.space}를 꾸미는 데 어울리는 인테리어 소품이나 가구를 추천해줘. "
        f"가능하면 책상 위/근처에 둘 수 있는 작은 소품, 조명, 무드등, 러그, 수납함, 포스터 등을 우선 고려해."
    )

    results = retriever.query(
        query_text=query_text,
        filters=None,
        top_k=30,
    )

    if not results:
        print("[DEBUG][SpaceRAG] 검색 결과 없음 → 카테고리 추론 실패")
        return

    score_by_cat: Dict[str, float] = defaultdict(float)

    for r in results:
        cat = r.get("category_id")
        if not cat:
            continue
        sim = float(r.get("sim_score", 0.0))
        score_by_cat[cat] += sim

    if not score_by_cat:
        print("[DEBUG][SpaceRAG] category_id 없는 결과만 나와서 추론 실패")
        return

    best_cat, best_score = max(score_by_cat.items(), key=lambda kv: kv[1])

    if best_score < 1.0:
        print(f"[DEBUG][SpaceRAG] 유사도 합 {best_score:.3f} < 1.0 → 카테고리 세팅 스킵")
        return

    session_state.category = best_cat
    print(
        f"[DEBUG][SpaceRAG] 공간 '{session_state.space}' → 카테고리 '{best_cat}' "
        f"(score_sum={best_score:.3f})"
    )


def is_smalltalk(user_text: str, turn_state: ChatState) -> bool:
    text = user_text.strip()

    no_structured = (
        turn_state.category is None
        and not turn_state.moods
        and turn_state.price_min is None
        and turn_state.price_max is None
        and turn_state.space is None
    )

    shopping_keywords = [
        "추천", "사고 싶", "사고싶", "사고 싶은",
        "골라줘", "찾아줘", "검색해줘",
        "러그", "커튼", "조명", "침대", "수납장",
        "가구", "인테리어", "집 꾸미", "집꾸미", "꾸미고",
    ]
    has_shopping_kw = any(kw in text for kw in shopping_keywords)

    greeting_keywords = ["안녕", "하이", "반가워", "고마워", "뭐해", "누구야", "ㅎㅎ", "ㅋㅋ"]

    is_greeting_like = any(kw in text for kw in greeting_keywords)

    if no_structured and (not has_shopping_kw) and (is_greeting_like or len(text) <= 20):
        return True

    return False


def is_interior_related(user_text: str, session_state: ChatState) -> bool:
    """
    "집 꾸미고 싶다" 계열의 발화를 interior 세션 시작 신호로 볼지 여부
    """
    text = user_text.strip()

    if (
        session_state.category is not None
        or session_state.moods
        or session_state.price_min is not None
        or session_state.price_max is not None
        or session_state.space is not None
    ):
        return True

    interior_words = ["집 꾸미", "집꾸미", "인테리어", "가구", "러그", "조명", "침대"]
    has_interior_kw = any(kw in text for kw in interior_words)
    has_home_word = "집" in text or "방" in text or "공간" in text

    return has_interior_kw or has_home_word


def is_ready_for_recommendation(session_state: ChatState, user_text: str) -> bool:
    """
    실제 상품 추천(RAG)을 해도 될 정도로 정보가 모였는지 판단
    - 카테고리 or 무드 중 하나는 있어야 함
    - 그리고 '추천해달라'는 의도나, 예산 정보 등이 있으면 추천 모드 진입
    """
    if not session_state.moods and session_state.category is None:
        return False

    text = user_text.strip()
    trigger_words = [
        "추천", "골라줘", "찾아줘", "사고 싶", "사고싶",
        "뭘 사야", "어떤 걸 사야", "어떤걸 사야", "제품 좀", "상품 좀",
    ]
    has_explicit_trigger = any(w in text for w in trigger_words)

    has_budget = session_state.price_min is not None or session_state.price_max is not None

    return has_explicit_trigger or has_budget


def make_survey_signature(state: ChatState) -> str:
    missing_category = state.category is None
    missing_mood = not state.moods
    missing_budget = state.price_min is None and state.price_max is None
    missing_space = state.space is None

    return f"cat:{missing_category}|mood:{missing_mood}|budget:{missing_budget}|space:{missing_space}"


def build_llm_history_from_messages(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Streamlit message 리스트 → llm_core.chat() 에 넣을 (user, assistant) pair 리스트로 변환
    """
    pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None

    return pairs


# =========================
# 3. Streamlit용 RAGRetriever 캐시
# =========================

@st.cache_resource(show_spinner=True)
def get_retriever() -> RAGRetriever:
    return RAGRetriever()


# =========================
# 4. 한 턴 처리 로직 (핵심)
# =========================

def handle_turn(
    user_text: str,
    rag_top_k: int,
    recommend_top_n: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    한 턴의 사용자 입력을 받아
    - LLM 답변 텍스트
    - (있다면) 추천 상품 리스트
    를 반환한다.
    """
    retriever = get_retriever()
    ss = st.session_state

    # === Streamlit 세션 상태 준비 ===
    if "chat_state" not in ss:
        ss["chat_state"] = ChatState()
    if "messages" not in ss:
        ss["messages"] = []

    session_state: ChatState = ss["chat_state"]
    messages: List[Dict[str, Any]] = ss["messages"]

    # llm_core.chat() 에 들어갈 history
    llm_history = build_llm_history_from_messages(messages)

    # 1) 이번 턴 파싱
    parsed = parse_user_query(user_text)
    turn_state = state_from_parsed(parsed)

    # 2) 세션 상태에 누적 반영
    update_session_state(session_state, turn_state)

    # 2-1) 공간 정보 기반 RAG로 카테고리 힌트
    infer_category_from_space_rag(retriever, session_state)

    # 4) 인테리어 세션 여부
    interior_session = is_interior_related(user_text, session_state)

    # 5) 모드 결정
    if not interior_session and is_smalltalk(user_text, turn_state):
        session_state.mode = ChatMode.SMALLTALK
    elif interior_session and not is_ready_for_recommendation(session_state, user_text):
        session_state.mode = ChatMode.SURVEY
    else:
        session_state.mode = ChatMode.RECOMMEND

    # === 모드별 처리 ===
    # SMALLTALK
    if session_state.mode == ChatMode.SMALLTALK:
        assistant_text = chat(
            history=llm_history,
            user_input=user_text,
            system_prompt=(
                "너는 친근한 한국어 AI 어시스턴트다. "
                "사용자가 인사하거나 잡담을 하면 자연스럽게 응답하되, "
                "집 꾸미기나 인테리어에 관심이 있는지 부드럽게 한 번 정도 물어봐도 좋다. "
                "다만 사용자가 원하지 않으면 억지로 쇼핑 얘기로 끌고 가지 마라."
            ),
            max_new_tokens=160,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return assistant_text, []

    # SURVEY (취향 질문)
    if session_state.mode == ChatMode.SURVEY:
        survey_sig = make_survey_signature(session_state)

        mood_question_prompt = (
            "사용자는 집을 꾸미고 싶어하지만, 아직 구체적인 취향 정보를 충분히 주지 않았다.\n"
            f"사용자 최근 발화: {user_text}\n\n"
            "너는 인테리어·홈데코 상담을 도와주는 어시스턴트다. "
            "아래 조건을 모두 만족하는 '질문만' 2~3문장으로 작성해라.\n"
            "1. 어떤 분위기/무드를 좋아하는지 물어본다. "
            "(예: 아늑한, 따뜻한, 미니멀, 우드톤, 호텔형, 북유럽, 화이트톤 등 예시 3~6개 정도를 가볍게 제시)\n"
            "2. 집에서 가장 먼저 꾸미고 싶은 공간이 어디인지 물어본다. "
            "(예: 거실, 침실, 작업실, 서재, 책상 근처 등 예시 포함)\n"
            "3. 대략적인 예산 범위를 물어본다. "
            "(예: 10만 원대, 30만 원 이하, 50만~100만 원 등)\n"
            "4. 사용자가 장난스럽거나 인테리어와 무관한 말을 해도, 그 내용은 깊게 반응하지 말고 "
            "다시 인테리어 취향(무드, 색, 공간, 예산)에 집중해서 질문해라.\n"
            "5. 이 단계에서는 상품 추천이나 브랜드 언급을 절대 하지 말고, 오직 질문만 해라."
        )

        # 이전과 동일한 설문 상태라면 표현 살짝 변경 요청
        if session_state.last_survey_signature == survey_sig:
            mood_question_prompt += (
                "\n6. 사용자가 이미 비슷한 질문을 한 번 들었을 수 있다. "
                "너무 똑같은 문장을 반복하기보다는, 표현을 조금 바꾸어 부담 없이 답할 수 있도록 부드럽게 물어봐라."
            )

        session_state.last_survey_signature = survey_sig

        assistant_text = chat(
            history=llm_history,
            user_input=mood_question_prompt,
            system_prompt=(
                "너는 인테리어·홈데코 상담을 도와주는 한국어 어시스턴트다. "
                "집을 꾸미고 싶다는 사용자가 나타나면, 먼저 취향(무드, 톤, 스타일), "
                "꾸미고 싶은 공간(거실, 침실 등), 대략적인 예산을 차근차근 물어봐야 한다. "
                "사용자가 엉뚱한 말을 하더라도, 인테리어 상담이라는 큰 흐름을 유지하도록 해라."
            ),
            max_new_tokens=220,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return assistant_text, []

    # RECOMMEND (상품 추천)
    rag_query_text = build_retrieval_query(user_text, session_state)
    filters = {"category_id": session_state.category} if session_state.category else None

    # ① 벡터 검색
    raw_results = retriever.query(
        query_text=rag_query_text,
        filters=filters,
        top_k=rag_top_k,
    )

    if not raw_results:
        assistant_text = chat(
            history=llm_history,
            user_input=(
                "사용자의 요청에 맞는 상품을 데이터베이스에서 찾지 못했다. "
                "이 사실을 부드럽게 설명하고, 다른 카테고리나 예산, "
                "혹은 원하는 무드를 조금 더 구체적으로 알려 달라고 안내해 줘.\n\n"
                f"사용자 요청: {user_text}"
            ),
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return assistant_text, []

    # ② 점수 기반 랭킹
    wrapped = [{"product": p} for p in raw_results]
    ranked = filter_and_rank(wrapped, session_state)

    if not ranked:
        assistant_text = chat(
            history=llm_history,
            user_input=(
                "벡터 검색 결과는 있었지만 필터링/랭킹 후에는 추천할 만한 상품이 없었다. "
                "이 사실을 부드럽게 설명하고, 예산 범위나 카테고리를 조금 넓혀서 "
                "다시 요청해 달라고 안내해 줘.\n\n"
                f"사용자 요청: {user_text}"
            ),
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return assistant_text, []

    top_ranked = ranked[:recommend_top_n]

    context_lines = []
    display_products: List[Dict[str, Any]] = []

    for idx, item in enumerate(top_ranked, start=1):
        p = item["product"]
        score = item["score"]

        pid = p.get("product_id", "")
        name = p.get("product_name", "")
        brand = p.get("brand_name", "")
        category = p.get("category_id", "")
        price = p.get("price", 0)
        moods = p.get("mood_keywords", []) or p.get("moods", [])
        link = p.get("link_url", "")
        img = p.get("image_url", "")

        context_lines.append(
            f"[추천 후보 {idx}]\n"
            f"  - product_id : {pid}\n"
            f"  - 상품명      : {name}\n"
            f"  - 브랜드      : {brand}\n"
            f"  - 카테고리    : {category}\n"
            f"  - 가격        : {format_won(price)}\n"
            f"  - 무드 키워드 : {', '.join(moods) if moods else 'N/A'}\n"
            f"  - 링크        : {link}\n"
            f"  - 이미지      : {img}\n"
            f"  - 랭킹 점수   : {score}\n"
        )

        display_products.append(
            {
                "rank": idx,
                "product_id": pid,
                "product_name": name,
                "brand_name": brand,
                "category_id": category,
                "price": price,
                "price_str": format_won(price),
                "moods": moods,
                "link_url": link,
                "image_url": img,
                "score": score,  # 내부적으로만 사용, UI에는 표시 안 함
            }
        )

    context_text = "\n".join(context_lines)

    llm_user_prompt = (
        "아래는 벡터 검색과 점수 기반 랭킹을 통해 고른 추천 후보 상품 목록이다.\n"
        "사용자의 요청과 아래 후보들을 참고해서, 한국어로 자연스럽게 추천 답변을 작성해라.\n\n"
        "⚠️ 매우 중요한 규칙:\n"
        "1. 반드시 아래 후보 목록에 있는 상품만 추천해야 한다. 목록에 없는 새 상품 이름을 만들거나, 새로운 제품을 상상해서 언급하면 안 된다.\n"
        "2. 추천할 때는 '추천 후보 1', '추천 후보 2', '추천 후보 3'처럼 **번호로만** 지칭하고, 상품명을 직접 다시 쓰지 마라.\n"
        "3. product_id, 상품명, 브랜드, 가격, 링크는 너가 말하지 말고, 시스템이 따로 보여준다고 생각해라.\n"
        "4. 실제로는 최대 3개까지만 추천해라. (가장 점수가 높은 2~3개만 선택해서 소개할 것)\n"
        "5. 각 추천 후보에 대해, 왜 사용자의 무드/카테고리/예산과 잘 맞는지 1~2문장 정도로 설명해라.\n"
        "6. 전체 답변은 최대 8문장 이내로 간결하게 작성하고, 문장을 중간에 끊지 말고 자연스럽게 끝까지 마무리해라.\n\n"
        "예시 형식:\n"
        "- 추천 후보 2는 현대적인 무드에 잘 맞고, 책상 근처에 두기 좋은 크기라서 추천해요.\n"
        "- 예산 3만 원 이내라면 추천 후보 2와 3을 우선 고려해 보시면 좋겠습니다.\n\n"
        f"=== 사용자 원문 요청 ===\n{user_text}\n\n"
        f"=== 추천 후보 상품 목록 ===\n{context_text}\n"
    )

    assistant_text = chat(
        history=llm_history,
        user_input=llm_user_prompt,
        max_new_tokens=260,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    return assistant_text, display_products


# =========================
# 5. 메인 Streamlit UI
# =========================

st.title("🛋️ 감성 기반 인테리어 상품 추천 데모")
st.caption("Qwen2.5-14B-Korean + Chroma RAG + Streamlit (로컬 실행)")

# --- 세션 초기화 ---
if "chat_state" not in st.session_state:
    st.session_state["chat_state"] = ChatState()
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "안녕하세요! 🐾 방 분위기를 바꾸고 싶은 인테리어 도우미입니다.\n\n"
                "예를 들어 이렇게 말해보세요:\n"
                "- `집이 전체적으로 너무 밋밋한데, 아늑한 우드톤 침실로 꾸미고 싶어요. 20만 원 안쪽이면 좋겠어요.`\n"
                "- `책상 근처에 둘 포인트 조명 추천해줘. 미니멀하고 화이트톤 느낌으로!`\n"
            ),
        }
    ]

messages: List[Dict[str, Any]] = st.session_state["messages"]


# --- 사이드바 (1): 프로젝트 정보 + 컨트롤 ---
with st.sidebar:
    st.header("프로젝트 정보")
    st.markdown(
        "- **목표**: 사용자의 방 사진/텍스트를 기반으로 무드/예산/공간에 맞는 인테리어 상품 추천\n"
        "- **LLM**: Qwen2.5-14B-Korean (8bit, 로컬)\n"
        "- **검색**: SentenceTransformer + ChromaDB RAG\n"
        "- **필터링**: 무드/카테고리/가격 기반 랭킹"
    )

    st.divider()
    st.subheader("RAG 파라미터")
    rag_top_k = st.slider("RAG 검색 결과 개수 (top_k)", 5, 40, RAG_TOP_K, 1)
    recommend_top_n = st.slider("표시할 추천 개수 (top_n)", 1, 5, RECOMMEND_TOP_N, 1)

    st.divider()
    if st.button("세션 상태 초기화", type="secondary"):
        st.session_state["chat_state"] = ChatState()
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "대화를 새로 시작해볼까요? 😊\n\n"
                    "예를 들어 이렇게 말해보세요:\n"
                    "- `아늑한 우드톤으로 침실 꾸미고 싶은데 20만 원 이하로 추천해줘`\n"
                    "- `책상 근처에 둘 포인트 조명 추천해줘. 미니멀한 화이트톤이면 좋겠어`"
                ),
            }
        ]
        st.success("대화 상태가 초기화되었습니다. (LLM / VectorDB는 그대로 유지)")
        st.experimental_rerun()


# --- 지금까지 메시지 렌더링 ---
for msg in messages:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    products = msg.get("products", [])

    with st.chat_message(role):
        st.markdown(content)
        # 추천 결과가 있는 assistant 메시지라면 카드로 함께 표시
        if role == "assistant" and products:
            st.markdown("---")
            st.markdown("**이번 추천에 사용된 상품 후보들:**")
            for p in products:
                with st.container():
                    cols = st.columns([1, 2])
                    with cols[0]:
                        if p.get("image_url"):
                            st.image(p["image_url"], use_container_width=True)
                        else:
                            st.write("이미지 없음")
                    with cols[1]:
                        st.markdown(f"**추천 후보 {p['rank']}**")
                        st.markdown(f"- 상품명: `{p['product_name']}`")
                        st.markdown(f"- 브랜드: `{p['brand_name']}`")
                        st.markdown(f"- 카테고리: `{p['category_id']}`")
                        st.markdown(f"- 가격: **{p['price_str']}**")
                        if p.get("moods"):
                            st.markdown(f"- 무드: {', '.join(p['moods'])}")
                        if p.get("link_url"):
                            st.markdown(f"- [상품 링크 열기]({p['link_url']})")


# --- 사용자 입력 ---
user_input = st.chat_input("방 분위기, 예산, 원하는 무드를 자유롭게 적어보세요!")

if user_input:
    # 1) 화면에 유저 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 한 턴 처리
    with st.chat_message("assistant"):
        with st.spinner("추천을 준비하고 있어요... (로컬 LLM + RAG 실행 중)"):
            assistant_text, products = handle_turn(
                user_input,
                rag_top_k=rag_top_k,
                recommend_top_n=recommend_top_n,
            )
        st.markdown(assistant_text)

        if products:
            st.markdown("---")
            st.markdown("**이번 추천에 사용된 상품 후보들:**")
            for p in products:
                with st.container():
                    cols = st.columns([1, 2])
                    with cols[0]:
                        if p.get("image_url"):
                            st.image(p["image_url"], use_container_width=True)
                        else:
                            st.write("이미지 없음")
                    with cols[1]:
                        st.markdown(f"**추천 후보 {p['rank']}**")
                        st.markdown(f"- 상품명: `{p['product_name']}`")
                        st.markdown(f"- 브랜드: `{p['brand_name']}`")
                        st.markdown(f"- 카테고리: `{p['category_id']}`")
                        st.markdown(f"- 가격: **{p['price_str']}**")
                        if p.get("moods"):
                            st.markdown(f"- 무드: {', '.join(p['moods'])}")
                        if p.get("link_url"):
                            st.markdown(f"- [상품 링크 열기]({p['link_url']})")

    # 3) 세션에 메시지 저장 (다음 턴을 위한 history)
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_text, "products": products}
    )


# --- 사이드바 (2): 세션 상태 요약 (항상 최신 chat_state 기준) ---
with st.sidebar:
    st.divider()
    st.subheader("세션 상태 (요약)")

    cs: ChatState = st.session_state["chat_state"]

    st.markdown(f"- **카테고리**: `{cs.category}`" if cs.category else "- **카테고리**: (미설정)")
    st.markdown(f"- **무드**: `{', '.join(cs.moods)}`" if cs.moods else "- **무드**: (미설정)")
    if cs.price_min is not None or cs.price_max is not None:
        st.markdown(
            f"- **예산**: "
            f"{format_won(cs.price_min) if cs.price_min is not None else '미정'} ~ "
            f"{format_won(cs.price_max) if cs.price_max is not None else '미정'}"
        )
    else:
        st.markdown("- **예산**: (미설정)")
    st.markdown(f"- **공간(space)**: `{cs.space}`" if cs.space else "- **공간(space)**: (미설정)")
    st.markdown(f"- **현재 모드**: `{cs.mode.name}`")
