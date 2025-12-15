import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time

# ────────────────────────────────────────────────
# 설정 (팀 환경에 맞게 수정하세요)
# ────────────────────────────────────────────────
EMBED_MODEL = "jhgan/ko-sroberta-multitask"  # 팀의 임베딩 모델로 변경
INDEX_PATH = "embeddb/index_cpu.faiss"  # 팀의 FAISS 인덱스 경로
META_PATH = "embeddb/metadata.jsonl"  # 팀의 메타데이터 경로

TOP_K = 10
NP = 64

# ────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────
@dataclass
class TestCase:
    """RAG 테스트 케이스"""
    query: str
    expected_product_ids: List[str]  # 정답으로 기대하는 상품 ID들
    expected_category: str = ""  # 기대하는 카테고리
    expected_brand: str = ""  # 기대하는 브랜드
    description: str = ""  # 테스트 케이스 설명

@dataclass
class EvalMetrics:
    """평가 지표"""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    category_match_rate: float
    brand_match_rate: float
    avg_latency_ms: float

# ────────────────────────────────────────────────
# 모델 & 인덱스 로드
# ────────────────────────────────────────────────
print("🔄 모델 및 인덱스 로드 중...")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
index = faiss.read_index(INDEX_PATH)
index.nprobe = NP

with open(META_PATH, encoding="utf-8") as f:
    metas = [json.loads(line) for line in f]

print(f"✅ 로드 완료: {index.ntotal} 상품, nprobe={NP}")
print(f"   메타데이터: {len(metas)}개")

# ────────────────────────────────────────────────
# RAG 검색 함수 (Bi-Encoder만 사용)
# ────────────────────────────────────────────────
def search(query: str, top_k: int = TOP_K):
    """쿼리에 대한 상품 검색 (FAISS만 사용)"""
    start_time = time.time()
    
    # 1) 쿼리 임베딩
    qv = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    
    # 2) FAISS 검색
    D, I = index.search(qv, top_k)
    
    # 3) 결과 구성
    results = []
    for idx, score in zip(I[0], D[0]):
        meta = metas[idx].copy()
        results.append((float(score), meta))
    
    latency_ms = (time.time() - start_time) * 1000
    return results, latency_ms

# ────────────────────────────────────────────────
# 평가 지표 계산 함수들
# ────────────────────────────────────────────────
def calculate_precision_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """Precision@K 계산"""
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    if len(retrieved_k) == 0:
        return 0.0
    return len(retrieved_k & expected_set) / len(retrieved_k)

def calculate_recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """Recall@K 계산"""
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)
    if len(expected_set) == 0:
        return 0.0
    return len(retrieved_k & expected_set) / len(expected_set)

def calculate_mrr(retrieved: List[str], expected: List[str]) -> float:
    """Mean Reciprocal Rank 계산"""
    expected_set = set(expected)
    for rank, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return 1.0 / rank
    return 0.0

def calculate_ndcg_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    """NDCG@K 계산"""
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in expected:
            dcg += 1.0 / np.log2(i + 1)
    
    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(expected), k)))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_category_match_rate(retrieved_metas: List[Dict], expected_category: str) -> float:
    """카테고리 매칭률 계산"""
    if not expected_category:
        return 1.0
    
    matches = sum(1 for meta in retrieved_metas if meta.get('category_id') == expected_category)
    return matches / len(retrieved_metas) if retrieved_metas else 0.0

def calculate_brand_match_rate(retrieved_metas: List[Dict], expected_brand: str) -> float:
    """브랜드 매칭률 계산"""
    if not expected_brand:
        return 1.0
    
    matches = sum(1 for meta in retrieved_metas if meta.get('brand_name') == expected_brand)
    return matches / len(retrieved_metas) if retrieved_metas else 0.0

# ────────────────────────────────────────────────
# 테스트 케이스 정의 (실제 데이터로 교체하세요)
# ────────────────────────────────────────────────
TEST_CASES = [
    TestCase(
        query="따뜻하고 아늑한 느낌의 거실 조명",
        expected_product_ids=["prod_001", "prod_045", "prod_023"],
        expected_category="조명",
        expected_brand="",
        description="따뜻한 무드 조명 추천"
    ),
    TestCase(
        query="모던하고 미니멀한 책상 소품",
        expected_product_ids=["prod_012", "prod_089", "prod_034"],
        expected_category="데스크 소품",
        expected_brand="",
        description="미니멀 데스크 소품"
    ),
    TestCase(
        query="빈티지 감성의 벽 장식",
        expected_product_ids=["prod_056", "prod_023", "prod_012"],
        expected_category="벽 장식",
        expected_brand="",
        description="빈티지 벽 데코"
    ),
    TestCase(
        query="북유럽 스타일 거실 러그",
        expected_product_ids=["prod_078", "prod_091", "prod_102"],
        expected_category="러그/카펫",
        expected_brand="",
        description="북유럽 스타일 러그"
    ),
    TestCase(
        query="화이트 톤 수납 바구니",
        expected_product_ids=["prod_134", "prod_156", "prod_189"],
        expected_category="수납/정리",
        expected_brand="",
        description="화이트 수납 바구니"
    ),
]

# ────────────────────────────────────────────────
# 평가 실행
# ────────────────────────────────────────────────
def evaluate_rag(test_cases: List[TestCase], k: int = TOP_K) -> Tuple[Dict, List[EvalMetrics]]:
    """RAG 시스템 전체 평가"""
    print(f"\n{'='*80}")
    print(f"🧪 RAG 성능 평가 시작 (총 {len(test_cases)}개 테스트 케이스)")
    print(f"   평가 방식: Bi-Encoder (FAISS) 단독 평가")
    print(f"{'='*80}\n")
    
    all_metrics = []
    latencies = []
    
    for i, tc in enumerate(test_cases, start=1):
        print(f"[{i}/{len(test_cases)}] {tc.description}")
        print(f"   Query: {tc.query}")
        
        # 검색 수행
        results, latency = search(tc.query, top_k=k)
        latencies.append(latency)
        
        # 결과 추출
        retrieved_ids = [meta['product_id'] for _, meta in results]
        retrieved_metas = [meta for _, meta in results]
        
        # 지표 계산
        precision = calculate_precision_at_k(retrieved_ids, tc.expected_product_ids, k)
        recall = calculate_recall_at_k(retrieved_ids, tc.expected_product_ids, k)
        mrr = calculate_mrr(retrieved_ids, tc.expected_product_ids)
        ndcg = calculate_ndcg_at_k(retrieved_ids, tc.expected_product_ids, k)
        category_match = calculate_category_match_rate(retrieved_metas, tc.expected_category)
        brand_match = calculate_brand_match_rate(retrieved_metas, tc.expected_brand)
        
        metrics = EvalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg_at_k=ndcg,
            category_match_rate=category_match,
            brand_match_rate=brand_match,
            avg_latency_ms=latency
        )
        all_metrics.append(metrics)
        
        print(f"   ✓ Precision@{k}: {precision:.3f}")
        print(f"   ✓ Recall@{k}: {recall:.3f}")
        print(f"   ✓ MRR: {mrr:.3f}")
        print(f"   ✓ NDCG@{k}: {ndcg:.3f}")
        print(f"   ✓ Category Match: {category_match:.3f}")
        if tc.expected_brand:
            print(f"   ✓ Brand Match: {brand_match:.3f}")
        print(f"   ✓ Latency: {latency:.1f}ms\n")
    
    # 전체 평균 계산
    avg_metrics = {
        'precision_at_k': np.mean([m.precision_at_k for m in all_metrics]),
        'recall_at_k': np.mean([m.recall_at_k for m in all_metrics]),
        'mrr': np.mean([m.mrr for m in all_metrics]),
        'ndcg_at_k': np.mean([m.ndcg_at_k for m in all_metrics]),
        'category_match_rate': np.mean([m.category_match_rate for m in all_metrics]),
        'brand_match_rate': np.mean([m.brand_match_rate for m in all_metrics]),
        'avg_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }
    
    return avg_metrics, all_metrics

# ────────────────────────────────────────────────
# 결과 출력
# ────────────────────────────────────────────────
def print_evaluation_report(avg_metrics: Dict, all_metrics: List[EvalMetrics]):
    """평가 결과 리포트 출력"""
    print(f"\n{'='*80}")
    print("📊 RAG 성능 평가 결과 종합")
    print(f"{'='*80}\n")
    
    print("【 검색 정확도 지표 】")
    print(f"  Precision@{TOP_K}      : {avg_metrics['precision_at_k']:.3f}")
    print(f"  Recall@{TOP_K}         : {avg_metrics['recall_at_k']:.3f}")
    print(f"  MRR                    : {avg_metrics['mrr']:.3f}")
    print(f"  NDCG@{TOP_K}           : {avg_metrics['ndcg_at_k']:.3f}")
    
    print("\n【 도메인 특화 지표 】")
    print(f"  Category Match Rate    : {avg_metrics['category_match_rate']:.3f}")
    print(f"  Brand Match Rate       : {avg_metrics['brand_match_rate']:.3f}")
    
    print("\n【 성능 지표 】")
    print(f"  평균 지연시간          : {avg_metrics['avg_latency_ms']:.1f}ms")
    print(f"  P95 지연시간           : {avg_metrics['p95_latency_ms']:.1f}ms")
    print(f"  P99 지연시간           : {avg_metrics['p99_latency_ms']:.1f}ms")
    
    print("\n【 성능 등급 】")
    score = (avg_metrics['precision_at_k'] * 0.3 + 
             avg_metrics['ndcg_at_k'] * 0.3 + 
             avg_metrics['category_match_rate'] * 0.4)
    
    if score >= 0.8:
        grade = "A (우수)"
    elif score >= 0.6:
        grade = "B (양호)"
    elif score >= 0.4:
        grade = "C (보통)"
    else:
        grade = "D (개선 필요)"
    
    print(f"  종합 점수             : {score:.3f}")
    print(f"  등급                  : {grade}")
    
    # 개선 제안
    print("\n【 개선 제안 】")
    if avg_metrics['precision_at_k'] < 0.5:
        print("  ⚠️  Precision이 낮습니다. 리랭크 모델 추가를 고려하세요.")
    if avg_metrics['category_match_rate'] < 0.7:
        print("  ⚠️  카테고리 매칭률이 낮습니다. 메타데이터에 카테고리 정보를 강화하세요.")
    if avg_metrics['avg_latency_ms'] > 100:
        print("  ⚠️  지연시간이 깁니다. nprobe 값을 줄이거나 인덱스를 최적화하세요.")
    if score >= 0.7:
        print("  ✅ 현재 성능이 우수합니다!")
    
    print(f"\n{'='*80}\n")

# ────────────────────────────────────────────────
# 상세 분석 함수
# ────────────────────────────────────────────────
def analyze_failure_cases(test_cases: List[TestCase], k: int = TOP_K):
    """실패 케이스 상세 분석"""
    print("\n🔍 실패 케이스 상세 분석\n")
    
    failure_count = 0
    for i, tc in enumerate(test_cases, start=1):
        results, _ = search(tc.query, top_k=k)
        retrieved_ids = [meta['product_id'] for _, meta in results]
        retrieved_metas = [meta for _, meta in results]
        
        expected_set = set(tc.expected_product_ids)
        retrieved_set = set(retrieved_ids)
        
        missed = expected_set - retrieved_set
        wrong = retrieved_set - expected_set
        
        if missed or len(wrong) > k // 2:
            failure_count += 1
            print(f"[케이스 {i}] {tc.description}")
            print(f"  Query: {tc.query}")
            if missed:
                print(f"  ⚠️  놓친 상품: {list(missed)}")
            if wrong:
                print(f"  ⚠️  잘못 검색된 상품 수: {len(wrong)}/{k}")
            
            # 실제 검색된 상위 3개 상품 정보 출력
            print(f"  📋 실제 검색된 상위 3개:")
            for rank, (score, meta) in enumerate(results[:3], start=1):
                print(f"      {rank}. [{meta['product_id']}] {meta.get('product_name', 'N/A')}")
                print(f"         - Category: {meta.get('category_id', 'N/A')}")
                print(f"         - Brand: {meta.get('brand_name', 'N/A')}")
                print(f"         - Score: {score:.4f}")
            print()
    
    if failure_count == 0:
        print("✅ 모든 테스트 케이스가 성공적으로 통과했습니다!\n")
    else:
        print(f"총 {failure_count}/{len(test_cases)}개 케이스에서 문제가 발견되었습니다.\n")

# ────────────────────────────────────────────────
# 검색 결과 시각화
# ────────────────────────────────────────────────
def display_search_results(query: str, top_k: int = 5):
    """특정 쿼리의 검색 결과를 자세히 출력"""
    print(f"\n{'='*80}")
    print(f"🔍 검색 결과 상세 보기")
    print(f"{'='*80}")
    print(f"Query: {query}\n")
    
    results, latency = search(query, top_k=top_k)
    
    for rank, (score, meta) in enumerate(results, start=1):
        print(f"{rank}. [{meta['product_id']}] {meta.get('product_name', 'N/A')}")
        print(f"   Score: {score:.4f}")
        print(f"   Category: {meta.get('category_id', 'N/A')}")
        print(f"   Brand: {meta.get('brand_name', 'N/A')}")
        print(f"   Price: {meta.get('price', 'N/A')}원")
        if meta.get('link_url'):
            print(f"   URL: {meta['link_url']}")
        print()
    
    print(f"⏱️  검색 소요 시간: {latency:.1f}ms")
    print(f"{'='*80}\n")

# ────────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("🏠 무드 기반 인테리어 소품 추천 RAG 평가 시스템")
    print("   (Bi-Encoder 단독 평가)")
    print("=" * 80)
    
    # 옵션 1: 전체 평가 실행
    print("\n[옵션 선택]")
    print("1. 전체 테스트 케이스 평가")
    print("2. 단일 쿼리 테스트")
    print("3. 둘 다 실행")
    
    choice = input("\n선택 (1/2/3) [기본값: 3]: ").strip() or "3"
    
    if choice in ["1", "3"]:
        # 1. 전체 평가 실행
        avg_metrics, all_metrics = evaluate_rag(TEST_CASES, k=TOP_K)
        
        # 2. 결과 리포트 출력
        print_evaluation_report(avg_metrics, all_metrics)
        
        # 3. 실패 케이스 분석
        analyze_failure_cases(TEST_CASES, k=TOP_K)
        
        # 4. JSON으로 저장
        result_json = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'embed_model': EMBED_MODEL,
                'top_k': TOP_K,
                'nprobe': NP,
                'reranker': None  # 리랭크 사용 안 함
            },
            'metrics': avg_metrics,
            'test_cases_count': len(TEST_CASES)
        }
        
        with open('rag_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        
        print("💾 결과가 'rag_evaluation_results.json'에 저장되었습니다.")
    
    if choice in ["2", "3"]:
        # 옵션 2: 단일 쿼리 테스트
        print("\n" + "="*80)
        test_query = input("테스트할 검색어를 입력하세요: ").strip()
        if test_query:
            display_search_results(test_query, top_k=10)