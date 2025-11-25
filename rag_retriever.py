# rag_retriever.py

"""
ChromaDB 기반 RAG 검색기 (최종 안정 버전)
 - SentenceTransformer로 쿼리 임베딩 생성
 - ChromaDB에서 top_k 검색
 - metadata 필터(category_id 등) 지원
 - 🔹 Chroma distance → sim_score(0~1)로 변환해서 메타데이터에 포함
"""
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, VECTOR_DB_PATH


# ============================================================
# 1. ChromaDB 초기화
# ============================================================

# build_vector_db.py에서 생성한 경로와 동일해야 함
CHROMA_DIR = VECTOR_DB_PATH
COLLECTION_NAME = "products"  # build_vector_db.py에서 생성한 이름과 동일해야 함


class RAGRetriever:
    def __init__(self):
        # 1) Chroma 클라이언트
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # 2) 컬렉션 로드
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

        # 3) 임베딩 모델 로드 (인덱싱과 동일한 모델 사용)
        self.encoder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # ========================================================
    # 2. 쿼리 수행
    # ========================================================

    def query(self, query_text: str, filters=None, top_k: int = 20):
        """
        query_text: 사용자 메시지 및 대화 상태를 기반으로 만든 검색 문장
        filters: {"category_id": "...", ...} 형태
        top_k: 검색 결과 개수

        return: [metadata(dict), metadata(dict), ...]
                각 dict 안에는 sim_score(0~1) 추가
        """

        # 🔹 쿼리 문장을 임베딩
        query_vec = self.encoder.encode([query_text], normalize_embeddings=True)[0]

        if filters:
            where = filters
        else:
            where = None

        # 🔹 거리 정보까지 함께 가져오기
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        # results["metadatas"], results["distances"]는 2차원 리스트
        metadatas_list = results.get("metadatas", [[]])[0]
        distances_list = results.get("distances", [[]])[0]

        cleaned = []

        for m, d in zip(metadatas_list, distances_list):
            if not m:
                continue

            item = dict(m)

            # 🔹 distance(코사인 거리) → 유사도(0~1)로 변환
            try:
                dist = float(d)
                sim = 1.0 - dist  # cosine distance이므로 1 - dist
                # 안전하게 0~1 사이로 클램프
                sim = max(0.0, min(1.0, sim))
            except Exception:
                sim = 0.0

            item["sim_score"] = round(sim, 4)

            # price가 문자열이거나 float일 수 있으므로 정수 변환 시도
            if "price" in item:
                try:
                    item["price"] = int(item["price"])
                except Exception:
                    pass

            # mood_keywords는 build_vector_db에서 " || "로 join된 문자열
            if "mood_keywords" in item:
                if isinstance(item["mood_keywords"], str):
                    raw = item["mood_keywords"]
                    # [ ... ] 같은 괄호 제거 (혹시 리스트 형태 문자열로 들어온 경우)
                    raw = (
                        raw.replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                    )
                    # " || " 또는 "," 기준으로 분리
                    parts = []
                    for chunk in raw.split("||"):
                        parts.extend(chunk.split(","))

                    item["mood_keywords"] = [
                        p.strip() for p in parts if p.strip()
                    ]
                elif isinstance(item["mood_keywords"], list):
                    # 이미 리스트이면 그대로 사용 (공백 정리)
                    item["mood_keywords"] = [
                        str(p).strip() for p in item["mood_keywords"] if str(p).strip()
                    ]

            cleaned.append(item)

        return cleaned


# ============================================================
# 3. 단독 테스트용 실행
# ============================================================

if __name__ == "__main__":
    retriever = RAGRetriever()

    test_query = "아늑한 베이지톤 거실 러그 추천해줘"
    results = retriever.query(test_query, filters={"category_id": "러그_커튼"}, top_k=5)

    print("\n=== TEST RESULT ===")
    for r in results:
        print(
            f"{r.get('product_name')} / moods={r.get('mood_keywords')} "
            f"/ sim_score={r.get('sim_score')}"
        )
