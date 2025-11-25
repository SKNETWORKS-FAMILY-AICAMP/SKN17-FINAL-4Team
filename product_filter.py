# product_filter.py
"""
상품 필터링 + 점수 기반 랭킹 시스템 (벡터 유사도 반영 버전)

점수 구성 요소:
- 벡터 유사도(sim_score)      → RAG가 뽑아준 전체 분위기/무드/텍스트 유사도
- 무드 일치도 (키워드 수준 보정)
- 카테고리 일치도
- 가격대 적합도 (가까울수록 점수↑)

추천 점수 예시:
    final_score =
        2.5 * sim_score +
        2.0 * mood_score +
        1.5 * category_score +
        1.0 * price_score
"""

import math


# ============================================================
# 1) 무드 점수 (키워드 보정용)
# ============================================================

def score_mood_match(product_moods, target_moods):
    """
    product_moods: ["아늑한", "베이지톤"]
    target_moods: 사용자가 원하는 무드 리스트

    - 완전 일치: 높은 점수
    - 부분 일치: 0.3 점수

    👉 벡터 유사도(sim_score)가 이미 전체 분위기를 크게 잡기 때문에
       이 함수는 '보너스 가중치' 정도 느낌으로만 사용.
    """

    if not target_moods:
        return 0.0
    if not product_moods:
        return 0.0

    pm = set([m.strip() for m in product_moods if m])
    tm = set([m.strip() for m in target_moods if m])

    if not pm:
        return 0.0

    exact = len(pm & tm)
    partial = sum(1 for t in tm if any(t in p or p in t for p in pm)) - exact

    # 정규화
    total = len(tm)
    exact_score = exact / total
    partial_score = min(partial, total - exact) * 0.3 / total

    return round(exact_score + partial_score, 4)


# ============================================================
# 2) 가격 점수
# ============================================================

def score_price_match(price: int, min_price: int, max_price: int):
    """
    가격대 안이면 1점.
    범위를 벗어나면 거리에 따라 감점.

    예: 100,000원 예산일 때
        - 110,000원: 약한 감점
        - 200,000원: 큰 감점
    """

    if price <= 0:
        return 0.0
    if min_price is None or max_price is None:
        return 0.0

    if min_price <= price <= max_price:
        return 1.0

    # 범위 벗어난 경우 거리 기반 페널티
    if price < min_price:
        diff = min_price - price
    else:
        diff = price - max_price

    # 100,000원 차이면 점수 거의 0
    penalty = math.exp(-diff / 70000)

    return round(penalty, 4)


# ============================================================
# 3) 카테고리 점수
# ============================================================

def score_category_match(cat: str, target: str):
    if not target:
        return 0.0
    if not cat:
        return 0.0

    # 완전 일치
    if cat == target:
        return 1.0

    # 부분 일치: 예) "러그" vs "러그_커튼"
    if target in cat or cat in target:
        return 0.5

    return 0.0


# ============================================================
# 4) 메인 함수: 필터 + 랭킹
# ============================================================

def filter_and_rank(products: list, state) -> list:
    """
    products: [
        {
            "product": {...},     # Chroma 메타데이터
            "sim_score": 0.87,    # RAG 벡터 유사도 (0~1)
        },
        ...
    ]
    state: ChatState
    """
    ranked = []

    for item in products:
        p = item["product"]
        base_sim = float(item.get("sim_score", 0.0))

        moods = p.get("mood_keywords", []) or p.get("moods", []) or []
        category = p.get("category_id", "")
        try:
            price = int(p.get("price", 0))
        except Exception:
            price = 0

        mood_score = score_mood_match(moods, state.moods)
        cat_score = score_category_match(category, state.category)
        price_score = score_price_match(price, state.price_min, state.price_max)

        # 🔹 최종 점수: 벡터 유사도 중심 + 키워드 기반 보정
        final_score = (
            2.5 * base_sim +
            2.0 * mood_score +
            1.5 * cat_score +
            1.0 * price_score
        )

        ranked.append({
            "score": round(final_score, 4),
            "product": p,
            "sim_score": round(base_sim, 4),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
