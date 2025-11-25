# build_vector_db.py

import json
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

from config import EMBEDDING_MODEL_NAME, PRODUCTS_JSON_PATH, VECTOR_DB_PATH


def sanitize_metadata(item: dict) -> dict:
    safe = {}
    for k, v in item.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, list):
            safe[k] = " || ".join(map(str, v))  # mood_keywords 같은 것
        else:
            safe[k] = str(v)
    return safe


def build_vector_db():
    print("▶ Vector DB 빌드 시작")
    print(f"  - PRODUCTS_JSON_PATH = {PRODUCTS_JSON_PATH}")
    print(f"  - VECTOR_DB_PATH     = {VECTOR_DB_PATH}")
    print(f"  - EMBEDDING_MODEL    = {EMBEDDING_MODEL_NAME}")

    with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"  - 로드된 상품 개수: {len(items)}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    try:
        client.delete_collection("products")
        print("  - 기존 'products' 컬렉션 삭제")
    except Exception:
        print("  - 기존 'products' 컬렉션 없음 (무시)")

    collection = client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn,
    )

    print("  - 컬렉션에 문서 추가 중...")

    BATCH_SIZE = 128
    batch_ids, batch_docs, batch_metas = [], [], []

    for item in tqdm(items):
        pid = item["product_id"]
        mood_text = " ".join(item.get("mood_keywords", []))

        doc = (
            f"{item.get('brand_name', '')} "
            f"{item.get('product_name', '')} "
            f"{item.get('category_id', '')} "
            f"{mood_text}"
        )

        batch_ids.append(pid)
        batch_docs.append(doc)
        batch_metas.append(sanitize_metadata(item))

        if len(batch_ids) >= BATCH_SIZE:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
            )
            batch_ids, batch_docs, batch_metas = [], [], []

    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
        )

    print("✅ Vector DB 빌드 완료!")


if __name__ == "__main__":
    build_vector_db()
