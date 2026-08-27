import pytest
from src.evaluation.metrics import (
    compute_dcg_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr_at_k,
    compute_bootstrap_ci,
)

def test_ndcg_perfect_ranking():
    retrieved = ["doc_A", "doc_B", "doc_C"]
    gold_qrels = {"doc_A": 2, "doc_B": 1, "doc_C": 0}
    ndcg = compute_ndcg_at_k(retrieved, gold_qrels, k=3)
    assert pytest.approx(ndcg, rel=1e-4) == 1.0

def test_ndcg_reversed_ranking():
    retrieved = ["doc_C", "doc_B", "doc_A"]
    gold_qrels = {"doc_A": 2, "doc_B": 1, "doc_C": 0}
    ndcg = compute_ndcg_at_k(retrieved, gold_qrels, k=3)
    assert 0.0 < ndcg < 1.0

def test_recall_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    gold_qrels = {"doc_2": 2, "doc_4": 1, "doc_99": 2}
    rec5 = compute_recall_at_k(retrieved, gold_qrels, k=5)
    # 2 out of 3 retrieved in top 5
    assert pytest.approx(rec5, rel=1e-4) == 2.0 / 3.0

def test_mrr_at_k():
    retrieved = ["doc_0", "doc_1", "doc_target", "doc_3"]
    gold_qrels = {"doc_target": 2}
    # doc_target is at rank 3
    mrr = compute_mrr_at_k(retrieved, gold_qrels, k=5)
    assert pytest.approx(mrr, rel=1e-4) == 1.0 / 3.0

def test_bootstrap_ci():
    values = [0.8, 0.85, 0.9, 0.75, 0.88, 0.92, 0.81]
    mean_val, lower, upper = compute_bootstrap_ci(values, n_bootstrap=500)
    assert lower <= mean_val <= upper

