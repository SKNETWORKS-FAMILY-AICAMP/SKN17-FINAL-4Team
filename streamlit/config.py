# config.py

"""
전역 설정 파일
- 모델명 / 경로
- 데이터 / 벡터DB 경로
- 추천 관련 기본 파라미터
"""

from pathlib import Path

# 현재 config.py 파일이 있는 디렉토리 (Path 객체)
BASE_DIR = Path(__file__).resolve().parent

# 같은 디렉토리에 있는 "model" 폴더 경로
HF_QWEN_MODEL_NAME = "MyeongHo0621/Qwen2.5-14B-Korean"
# HF_QWEN_MODEL_NAME = BASE_DIR / "model" <<< 이거로 써도 됨

DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# =========================
# 모델 설정
# =========================

# RAG 임베딩 모델 (SentenceTransformers)
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean"

# =========================
# 데이터 경로
# =========================

PRODUCTS_JSON_PATH = str(DATA_DIR / "products_all_ver1.json")
VECTOR_DB_PATH = str(VECTOR_DB_DIR)

# =========================
# 추천 시스템 기본 설정
# =========================

RAG_TOP_K = 20          # 벡터 검색 상위 K개
RECOMMEND_TOP_N = 3     # 사용자에게 보여줄 추천 개수
PRICE_TOLERANCE = 1.15  # (필요하면 예산 범위 넓힐 때 사용)
