# main.py
"""
Qwen2.5-14B-Korean + Chroma RAG 기반
대화형 상품 추천 CLI 챗봇 (간단 상태머신 기반)

Flow:
1) 사용자 입력 받기
2) parse_user_query()로 카테고리/무드/예산/공간(space) 추출 (턴 단위)
3) 세션 상태(session_state)에 누적 반영
4) 공간(space)이 있는데 카테고리가 없으면 RAG로 category 힌트 추론
5) 모드 결정:
   - SMALLTALK : 잡담 모드
   - SURVEY    : 취향/공간/예산 질문 모드
   - RECOMMEND : RAG + 필터링 + 상품 추천 모드
6) 각 모드별로 LLM 호출
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any

from config import (
    RAG_TOP_K,
    RECOMMEND_TOP_N,
)
from rag_retriever import RAGRetriever
from product_filter import filter_and_rank
from llm_core import chat, parse_user_query


# =========================
# 0. 모드 정의 (상태 머신용)
# =========================

class ChatMode(Enum):
    SMALLTALK = auto()   # 잡담 모드
    SURVEY = auto()      # 취향/공간/예산 질문 모드
    RECOMMEND = auto()   # 상품 추천 모드


# =========================
# 1. 상태 정의
# =========================

@dataclass
class ChatState:
    # 사용자가 지금까지 제공한 선호 정보(세션 누적)
    category: Optional[str] = None
    moods: List[str] = field(default_factory=list)
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    space: Optional[str] = None  # 🔹 새로 추가: 꾸미고 싶은 공간(책상 근처, 침실, 거실 등)

    # 상태머신 관련
    mode: ChatMode = ChatMode.SMALLTALK

    # 반복 질문 완화용: 마지막 설문 상태 시그니처
    last_survey_signature: Optional[str] = None


# =========================
# 2. 유틸 함수
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
        parts.append(f"꾸미고 싶은 공간: {state.space}")  # 🔹 공간도 힌트로

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

    # 🔹 공간 정보 누적
    if turn_state.space:
        session_state.space = turn_state.space


# 🔹 공간 → 카테고리 힌트 (RAG 기반)
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

    # 너무 애매한 경우는 세팅하지 않도록 간단한 임계값
    # (top_k=30 기준으로, 합이 1.0 이하면 정보가 약하다고 보고 스킵)
    if best_score < 1.0:
        print(f"[DEBUG][SpaceRAG] 유사도 합 {best_score:.3f} < 1.0 → 카테고리 세팅 스킵")
        return

    session_state.category = best_cat
    print(
        f"[DEBUG][SpaceRAG] 공간 '{session_state.space}' → 카테고리 '{best_cat}' "
        f"(score_sum={best_score:.3f})"
    )


# 🔹 인사/잡담 여부 판단 (Heuristic + LLM 파서 결과)
def is_smalltalk(user_text: str, turn_state: ChatState) -> bool:
    text = user_text.strip()

    # 파서 기준: 이번 턴에서 구조화 정보가 거의 없음
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

    # 아직 아무 정보도 없고 / 쇼핑 관련도 아니고 / 인사 같거나 짧은 문장 → 그냥 스몰톡
    if no_structured and (not has_shopping_kw) and (is_greeting_like or len(text) <= 20):
        return True

    return False


def is_interior_related(user_text: str, session_state: ChatState) -> bool:
    """
    "집 꾸미고 싶다" 계열의 발화를 interior 세션 시작 신호로 볼지 여부
    (스몰톡과의 경계는 애매해서, 일단 '인테리어 관련 단어 + 집/방/공간' 정도로 판단)
    """
    text = user_text.strip()

    # 이미 세션 상태에 뭔가 쌓여 있다면 인테리어 세션으로 간주
    if (
        session_state.category is not None
        or session_state.moods
        or session_state.price_min is not None
        or session_state.price_max is not None
        or session_state.space is not None     # 🔹 공간 정보만 있어도 인테리어 세션으로 본다
    ):
        return True

    # 텍스트 기반 힌트
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
    # 기본적으로 아무 것도 없으면 추천 X
    if not session_state.moods and session_state.category is None:
        return False

    text = user_text.strip()
    trigger_words = [
        "추천", "골라줘", "찾아줘", "사고 싶", "사고싶",
        "뭘 사야", "어떤 걸 사야", "어떤걸 사야", "제품 좀", "상품 좀",
    ]
    has_explicit_trigger = any(w in text for w in trigger_words)

    # 예산이 있다면, 추천해도 무방하다고 판단
    has_budget = session_state.price_min is not None or session_state.price_max is not None

    return has_explicit_trigger or has_budget


def make_survey_signature(state: ChatState) -> str:
    """
    설문 상태 시그니처: 어떤 정보가 비어 있는지에 따라 생성
    (같은 시그니처에서 계속 설문이면, 반복 질문 완화에 활용 가능)
    """
    missing_category = state.category is None
    missing_mood = not state.moods
    missing_budget = state.price_min is None and state.price_max is None
    missing_space = state.space is None

    return f"cat:{missing_category}|mood:{missing_mood}|budget:{missing_budget}|space:{missing_space}"


# =========================
# 3. 메인 루프
# =========================

def main():
    retriever = RAGRetriever()
    history: List[Tuple[str, str]] = []

    # 세션 전체에 공유되는 누적 상태
    session_state = ChatState()

    print("===============================================")
    print("  감성 기반 상품 추천 챗봇 (RAG + Qwen2.5-14B-Korean)")
    print("   - 종료하려면 'exit' 또는 'quit' 입력")
    print("===============================================")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[시스템] 종료합니다.")
            break

        if user_text.lower() in {"exit", "quit"}:
            print("[시스템] 종료합니다.")
            break

        # 1) 이번 턴 파싱 (LLM 파서)
        parsed = parse_user_query(user_text)
        turn_state = state_from_parsed(parsed)

        # 2) 세션 상태에 누적 반영
        update_session_state(session_state, turn_state)

        # 2-1) 공간 정보 기반 RAG 임베딩으로 카테고리 힌트 추론
        infer_category_from_space_rag(retriever, session_state)

        # 3) 디버그: 이번 턴 vs 세션 누적 상태 출력
        print("\n[DEBUG] 이번 턴 파싱 결과:")
        print(f"  - category : {turn_state.category}")
        print(f"  - moods    : {turn_state.moods}")
        print(f"  - price_min: {turn_state.price_min}")
        print(f"  - price_max: {turn_state.price_max}")
        print(f"  - space    : {turn_state.space}")

        print("[DEBUG] 세션 누적 상태:")
        print(f"  - category : {session_state.category}")
        print(f"  - moods    : {session_state.moods}")
        print(f"  - price_min: {session_state.price_min}")
        print(f"  - price_max: {session_state.price_max}")
        print(f"  - space    : {session_state.space}")

        # 4) 인테리어 세션 여부 판단
        interior_session = is_interior_related(user_text, session_state)

        # 5) 모드 결정 (상태 머신의 전이 규칙)
        if not interior_session and is_smalltalk(user_text, turn_state):
            session_state.mode = ChatMode.SMALLTALK
        elif interior_session and not is_ready_for_recommendation(session_state, user_text):
            session_state.mode = ChatMode.SURVEY
        else:
            session_state.mode = ChatMode.RECOMMEND

        # 6) 디버그: 현재 모드 출력
        print(f"[DEBUG] 현재 모드: {session_state.mode.name}")

        # =========================
        # ① SMALLTALK MODE
        # =========================
        if session_state.mode == ChatMode.SMALLTALK:
            assistant_text = chat(
                history=history,
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
            print(f"\nAssistant:\n{assistant_text}\n")
            history.append((user_text, assistant_text))
            continue

        # =========================
        # ② SURVEY MODE (취향 질문)
        # =========================
        if session_state.mode == ChatMode.SURVEY:
            # 반복 질문 완화용 시그니처 계산
            survey_sig = make_survey_signature(session_state)

            # 기본 프롬프트
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

            # 이전과 동일한 설문 상태라면, 살짝 다른 표현을 쓰도록 추가 힌트
            if session_state.last_survey_signature == survey_sig:
                mood_question_prompt += (
                    "\n6. 사용자가 이미 비슷한 질문을 한 번 들었을 수 있다. "
                    "너무 똑같은 문장을 반복하기보다는, 표현을 조금 바꾸어 부담 없이 답할 수 있도록 부드럽게 물어봐라."
                )

            session_state.last_survey_signature = survey_sig

            assistant_text = chat(
                history=history,
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

            print(f"\nAssistant:\n{assistant_text}\n")
            history.append((user_text, assistant_text))
            continue

        # =========================
        # ③ RECOMMEND MODE (실제 추천)
        # =========================
        # 여기까지 왔다는 것은:
        # - 인테리어 세션이 시작되었고,
        # - 무드/카테고리/예산 중 최소 하나는 채워져 있음 → 실제 추천 진행
        rag_query_text = build_retrieval_query(user_text, session_state)

        # 카테고리 필터 (없으면 None)
        filters = {"category_id": session_state.category} if session_state.category else None

        # ③-1) 벡터 검색
        raw_results = retriever.query(
            query_text=rag_query_text,
            filters=filters,
            top_k=RAG_TOP_K,
        )

        if not raw_results:
            assistant_text = chat(
                history=history,
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
            print(f"\nAssistant:\n{assistant_text}\n")
            history.append((user_text, assistant_text))
            continue

        # ③-2) 점수 기반 랭킹
        wrapped = [{"product": p} for p in raw_results]
        ranked = filter_and_rank(wrapped, session_state)

        if not ranked:
            assistant_text = chat(
                history=history,
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
            print(f"\nAssistant:\n{assistant_text}\n")
            history.append((user_text, assistant_text))
            continue

        # ③-3) 상위 N개만 LLM에게 전달
        top_ranked = ranked[:RECOMMEND_TOP_N]

        context_lines = []
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

            line = (
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
            context_lines.append(line)

        context_text = "\n".join(context_lines)

        # 🔹 LLM에게는 "후보 번호만 써라"라고 시키기 (상품명/링크는 우리가 따로 출력)
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
            history=history,
            user_input=llm_user_prompt,
            max_new_tokens=260,   # 너무 길지 않게 제한
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # 🔹 출력: 자연어 설명 + 실제 RAG 결과(상품명/링크)는 우리가 직접 출력
        print(f"\nAssistant:\n{assistant_text}\n")
        print("[SYSTEM] 아래는 방금 추천에 사용된 실제 상품 정보입니다.")
        print(context_text)
        print()

        history.append((user_text, assistant_text))


if __name__ == "__main__":
    main()
