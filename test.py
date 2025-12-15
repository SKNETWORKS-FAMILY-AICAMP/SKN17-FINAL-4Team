import json
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 클래스"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    details: List[Dict]

class ChatbotEvaluator:
    """챗봇 성능 평가 클래스"""
    
    def __init__(self):
        self.results = []
    
    # ==================== 1. 상품 정답률 평가 ====================
    # 상품 정답률: 사용자가 요구한 조건에 부합한 상품을 찾아서 올바르게 추천하는가
    
    def evaluate_product_recommendations(
        self,
        test_cases: List[Dict],
        get_recommendations_func
    ) -> EvaluationResult:
        """
        상품 추천 정답률 평가
        
        Args:
            test_cases: 테스트 케이스 리스트
                [
                    {
                        "query": "20만원대 노트북 추천해줘",
                        "expected_products": ["prod_1", "prod_2", "prod_3"],
                        "conditions": {"price_range": [150000, 250000], "category": "노트북"}
                    },
                    ...
                ]
            get_recommendations_func: 챗봇의 추천 함수 (query를 받아서 추천 상품 ID 리스트 반환)
        
        Returns:
            EvaluationResult 객체
        """
        details = []
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        correct_count = 0
        
        for i, case in enumerate(test_cases):
            query = case["query"]
            expected = set(case["expected_products"])
            
            # 챗봇으로부터 추천 받기
            try:
                predicted = set(get_recommendations_func(query))
            except Exception as e:
                print(f"Error in case {i}: {e}")
                predicted = set()
            
            # 메트릭 계산(precision, recall, f1 score 측정을 위해 필요한 결과값들)
            tp = len(expected & predicted)  # True Positive
            fp = len(predicted - expected)  # False Positive
            fn = len(expected - predicted)  # False Negative
            
            precision = tp / len(predicted) if len(predicted) > 0 else 0
            recall = tp / len(expected) if len(expected) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 완전 일치 여부
            is_correct = expected == predicted
            if is_correct:
                correct_count += 1
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            # 상세 결과 저장
            details.append({
                "case_id": i,
                "query": query,
                "expected": list(expected),
                "predicted": list(predicted),
                "true_positives": list(expected & predicted),
                "false_positives": list(predicted - expected),
                "false_negatives": list(expected - predicted),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "is_correct": is_correct
            })
        
        n = len(test_cases)
        return EvaluationResult(
            precision=total_precision / n if n > 0 else 0,
            recall=total_recall / n if n > 0 else 0,
            f1_score=total_f1 / n if n > 0 else 0,
            accuracy=correct_count / n if n > 0 else 0,
            details=details
        )
    
    # ==================== 2. Retrieval 정확도 평가(RAG 성능 검증) ====================
    
    def evaluate_retrieval_accuracy(
        self,
        test_cases: List[Dict],
        retrieval_func,
        k: int = 5
    ) -> EvaluationResult:
        """
        RAG Retrieval 정확도 평가
        
        Args:
            test_cases: 테스트 케이스 리스트
                [
                    {
                        "query": "이 사진대로 방을 꾸미고 싶어요. 어떤 무드의 방으로 꾸미면 좋을까요?",
                        "relevant_docs": ["doc_15", "doc_23"],  # 관련 문서 ID
                        "retrieved_k": 5  # 상위 k개 검색 (옵션)
                    },
                    ...
                ]
            retrieval_func: RAG 검색 함수 (query와 k를 받아서 문서 ID 리스트 반환)
            k: 검색할 문서 개수 (기본값: 5)
        
        Returns:
            EvaluationResult 객체
        """
        details = []
        total_precision_at_k = 0
        total_recall_at_k = 0
        total_mrr = 0  # Mean Reciprocal Rank
        total_ndcg = 0  # Normalized Discounted Cumulative Gain
        
        for i, case in enumerate(test_cases):
            query = case["query"]
            relevant_docs = set(case["relevant_docs"])
            case_k = case.get("retrieved_k", k)
            
            # RAG 검색 수행
            try:
                retrieved = retrieval_func(query, case_k)
            except Exception as e:
                print(f"Error in case {i}: {e}")
                retrieved = []
            
            # Precision@K 및 Recall@K 계산
            retrieved_set = set(retrieved[:case_k])
            tp = len(relevant_docs & retrieved_set)
            
            precision_at_k = tp / case_k if case_k > 0 else 0
            recall_at_k = tp / len(relevant_docs) if len(relevant_docs) > 0 else 0
            
            # MRR (Mean Reciprocal Rank) 계산
            reciprocal_rank = 0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_docs:
                    reciprocal_rank = 1 / rank
                    break
            
            # NDCG 계산
            dcg = sum([1 / np.log2(rank + 1) for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant_docs])
            idcg = sum([1 / np.log2(rank + 1) for rank in range(1, min(len(relevant_docs), case_k) + 1)])
            ndcg = dcg / idcg if idcg > 0 else 0
            
            total_precision_at_k += precision_at_k
            total_recall_at_k += recall_at_k
            total_mrr += reciprocal_rank
            total_ndcg += ndcg
            
            # 상세 결과 저장
            details.append({
                "case_id": i,
                "query": query,
                "relevant_docs": list(relevant_docs),
                "retrieved_docs": retrieved,
                "hits": list(relevant_docs & retrieved_set),
                "precision@k": precision_at_k,
                "recall@k": recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                "ndcg": ndcg,
                "k": case_k
            })
        
        n = len(test_cases)
        return EvaluationResult(
            precision=total_precision_at_k / n if n > 0 else 0,
            recall=total_recall_at_k / n if n > 0 else 0,
            f1_score=2 * (total_precision_at_k / n) * (total_recall_at_k / n) / 
                     ((total_precision_at_k / n) + (total_recall_at_k / n)) 
                     if (total_precision_at_k + total_recall_at_k) > 0 else 0,
            accuracy=total_mrr / n if n > 0 else 0,  # MRR을 accuracy로 사용
            details=details
        )
    
    # ==================== 결과 출력 및 저장 ====================
    
    def print_results(self, result: EvaluationResult, eval_type: str = "Product"):
        """평가 결과 출력"""
        print(f"\n{'='*60}")
        print(f"{eval_type} Evaluation Results")
        print(f"{'='*60}")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall: {result.recall:.4f}")
        print(f"F1-Score: {result.f1_score:.4f}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"{'='*60}\n")
        
        # 실패 케이스 출력
        failed_cases = [d for d in result.details if not d.get("is_correct", True)]
        if failed_cases:
            print(f"Failed Cases: {len(failed_cases)}")
            for case in failed_cases[:3]:  # 처음 3개만 출력
                print(f"\n- Query: {case['query']}")
                if 'false_positives' in case:
                    print(f"  False Positives: {case['false_positives']}")
                    print(f"  False Negatives: {case['false_negatives']}")
    
    def save_results(self, result: EvaluationResult, filename: str):
        """평가 결과를 JSON 파일로 저장"""
        output = {
            "summary": {
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "accuracy": result.accuracy
            },
            "details": result.details
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {filename}")


# ==================== 사용 예시 ====================

if __name__ == "__main__":
    evaluator = ChatbotEvaluator()
    
    # 1. 상품 추천 평가 예시
    print("\n[1] 상품 추천 정답률 평가")
    print("-" * 60)
    
    # 테스트 케이스 정의
    product_test_cases = [
        {
            "query": "20만원대 노트북 추천해줘",
            "expected_products": ["laptop_001", "laptop_002", "laptop_005"],
            "conditions": {"price_range": [150000, 250000]}
        },
        {
            "query": "가벼운 무선 마우스 찾아줘",
            "expected_products": ["mouse_101", "mouse_103"],
            "conditions": {"weight": "light", "type": "wireless"}
        }
    ]
    
    # 더미 추천 함수 (실제로는 여러분의 챗봇 함수로 교체)
    def dummy_recommendation_func(query):
        if "노트북" in query:
            return ["laptop_001", "laptop_002", "laptop_007"]  # laptop_005 누락
        elif "마우스" in query:
            return ["mouse_101", "mouse_103", "mouse_105"]  # mouse_105 추가
        return []
    
    product_result = evaluator.evaluate_product_recommendations(
        test_cases=product_test_cases,
        get_recommendations_func=dummy_recommendation_func
    )
    
    evaluator.print_results(product_result, "Product Recommendation")
    
    
    # 2. RAG Retrieval 평가 예시
    print("\n[2] RAG Retrieval 정확도 평가")
    print("-" * 60)
    
    retrieval_test_cases = [
        {
            "query": "환불 정책이 어떻게 되나요?",
            "relevant_docs": ["doc_015", "doc_023"],
            "retrieved_k": 5
        },
        {
            "query": "배송 기간은 얼마나 걸리나요?",
            "relevant_docs": ["doc_042", "doc_088", "doc_091"],
            "retrieved_k": 5
        }
    ]
    
    # 더미 검색 함수 (실제로는 여러분의 RAG 함수로 교체)
    def dummy_retrieval_func(query, k):
        if "환불" in query:
            return ["doc_015", "doc_100", "doc_023", "doc_055", "doc_012"]
        elif "배송" in query:
            return ["doc_042", "doc_088", "doc_200", "doc_091", "doc_033"]
        return []
    
    retrieval_result = evaluator.evaluate_retrieval_accuracy(
        test_cases=retrieval_test_cases,
        retrieval_func=dummy_retrieval_func,
        k=5
    )
    
    evaluator.print_results(retrieval_result, "RAG Retrieval")
    
    # 결과 저장
    # evaluator.save_results(product_result, "product_eval_results.json")
    # evaluator.save_results(retrieval_result, "retrieval_eval_results.json")