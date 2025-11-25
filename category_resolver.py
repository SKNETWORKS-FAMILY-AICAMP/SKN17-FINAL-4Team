# category_resolver.py
"""
사용자 자연어 입력으로부터 category_id를
벡터 유사도 기반으로 추론하는 모듈.

- products_all_ver1.json 에서 실제 category_id 목록을 추출
- SentenceTransformer(EMBEDDING_MODEL_NAME)로
  카테고리 문장과 유저 입력을 임베딩
- 코사인 유사도 가장 높은 카테고리를 반환
"""

import json
from typing import Optional, List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, PRODUCTS_JSON_PATH


# 전역 캐시
_model: SentenceTransformer | None = None
_category_labels: List[str] | None = None
_category_vecs: np.ndarray | None = None


def _ensure_initialized():
    """
    - 임베딩 모델 로드
    - products_all_ver1.json에서 category_id 고유값 추출
    - 각 카테고리에 대한 임베딩 미리 계산
    """
    global _model, _category_labels, _category_vecs

    if _model is not None and _category_labels is not None and _category_vecs is not None:
        return

    print("🔎 [CategoryResolver] 초기화 중...")

    # 1) 모델 로드
    _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2) JSON에서 카테고리 목록 추출
    with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_set = set()
    for item in data:
        cid = item.get("category_id")
        if cid:
            category_set.add(cid)

    _category_labels = sorted(category_set)

    # 3) 카테고리 문장을 약간 풍부하게 만들어서 임베딩
    category_texts = [
        f"인테리어 상품 카테고리 {cid}"
        for cid in _category_labels
    ]

    _category_vecs = _model.encode(
        category_texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    print(f"🔎 [CategoryResolver] 카테고리 개수: {len(_category_labels)}개 초기화 완료")


def infer_category_from_text(
    user_text: str,
    min_similarity: float = 0.42,
) -> Optional[str]:
    """
    유저 자연어 입력을 받아서
    가장 유사한 category_id를 반환한다.

    - min_similarity: 이 값보다 낮으면 None (카테고리 추론 실패로 간주)
    """
    _ensure_initialized()

    assert _model is not None
    assert _category_labels is not None
    assert _category_vecs is not None

    text = user_text.strip()
    if not text:
        return None

    # 유저 입력 임베딩
    q_vec = _model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]  # shape: (dim,)

    # 코사인 유사도 = dot(normalized_vecs)
    sims = np.dot(_category_vecs, q_vec)  # shape: (num_categories,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    best_cat = _category_labels[best_idx]

    print(
        f"[CategoryResolver] best category = {best_cat} "
        f"(similarity={best_sim:.3f})"
    )

    if best_sim < min_similarity:
        # 신뢰도 낮으면 카테고리 안 잡음
        return None

    return best_cat
