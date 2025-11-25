# rag_index.py
"""
products_all_ver1.json → Chroma VectorDB 인덱싱 스크립트

- 한 상품당 하나의 document
- 무드 키워드 / 카테고리 / 가격 / 브랜드 등 메타데이터 저장
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    PRODUCTS_JSON_PATH,
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
)

def build_index():
    print("▶ RAG 인덱싱 시작")
    print(f"  - JSON: {PRODUCTS_JSON_PATH}")
    print(f"  - Vector DB: {VECTOR_DB_DIR}")

    # 경로 생성
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Chroma 클라이언트
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"},
    )

    # 임베딩 모델
    print(f"🧠 임베딩 모델 로딩 중... ({EMBEDDING_MODEL_NAME})")
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # JSON 로드
    with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
        products: List[Dict[str, Any]] = json.load(f)

    print(f"📦 총 상품 수: {len(products)}개")

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for p in products:
        product_id = p.get("product_id")
        category_id = p.get("category_id", "")
        brand_name = p.get("brand_name", "")
        product_name = p.get("product_name", "")
        price_str = p.get("price", "0")
        moods = p.get("mood_keywords", []) or []

        try:
            price_int = int(price_str)
        except Exception:
            price_int = 0

        # 임베딩용 텍스트 구성
        text_parts = [
            f"[카테고리] {category_id}",
            f"[브랜드] {brand_name}",
            f"[상품명] {product_name}",
            f"[가격] {price_str}원",
        ]
        if moods:
            text_parts.append("[무드 키워드] " + ", ".join(moods))

        # 나중에 OCR로 상품 설명 붙일 수 있음
        # description_text = p.get("description_text", "")
        # if description_text:
        #     text_parts.append("[상품 설명] " + description_text)

        doc_text = "\n".join(text_parts)

        ids.append(product_id)
        docs.append(doc_text)
        metadatas.append(
            {
                "product_id": product_id,
                "category_id": category_id,
                "brand_name": brand_name,
                "price": price_int,
                "moods": moods,
                "link_url": p.get("link_url", ""),
                "image_url": p.get("image_url", ""),
                "source_site": infer_source_site(product_id),
            }
        )

    print("🧠 임베딩 계산 중...")
    embeddings = emb_model.encode(docs, batch_size=64, show_progress_bar=True)

    print("💾 Chroma 컬렉션에 추가 중...")
    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("✅ 인덱싱 완료!")


def infer_source_site(product_id: str) -> str:
    """
    product_id 패턴으로 간단히 출처 분류
    예: ten_..., kakao_..., guud_...
    """
    if product_id.startswith("ten_"):
        return "10x10"
    if product_id.startswith("kakao_"):
        return "kakao"
    if product_id.startswith("guud_"):
        return "guud"
    return "unknown"


if __name__ == "__main__":
    build_index()
