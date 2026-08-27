from typing import List, Dict, Tuple, Any, Optional
import math
import numpy as np

def compute_dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at rank k with graded exponential gain:
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(i + 2)  # i is 0-indexed, so rank is i+1, log2(rank + 1) = log2(i + 2)
        dcg += gain / discount
    return dcg

def compute_ndcg_at_k(
    retrieved_page_ids: List[str],
    gold_qrels: Dict[str, int],  # mapping page_id -> relevance (0, 1, 2)
    k: int = 5,
) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG@k)."""
    if not gold_qrels or all(v == 0 for v in gold_qrels.values()):
        return 0.0

    actual_rels = [gold_qrels.get(pid, 0) for pid in retrieved_page_ids[:k]]
    actual_dcg = compute_dcg_at_k(actual_rels, k)

    # Ideal DCG: sort all positive qrels in descending order
    ideal_rels = sorted(list(gold_qrels.values()), reverse=True)[:k]
    ideal_dcg = compute_dcg_at_k(ideal_rels, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_recall_at_k(
    retrieved_page_ids: List[str],
    gold_qrels: Dict[str, int],
    k: int = 5,
    min_rel: int = 1,
) -> float:
    """Computes Recall@k: fraction of relevant pages (rel >= min_rel) retrieved in top k."""
    relevant_targets = {pid for pid, rel in gold_qrels.items() if rel >= min_rel}
    if not relevant_targets:
        return 0.0

    retrieved_set = set(retrieved_page_ids[:k])
    hits = len(relevant_targets.intersection(retrieved_set))
    return hits / len(relevant_targets)

def compute_mrr_at_k(
    retrieved_page_ids: List[str],
    gold_qrels: Dict[str, int],
    k: int = 10,
    min_rel: int = 1,
) -> float:
    """Computes Mean Reciprocal Rank (MRR@k)."""
    for rank, pid in enumerate(retrieved_page_ids[:k], start=1):
        if gold_qrels.get(pid, 0) >= min_rel:
            return 1.0 / rank
    return 0.0

def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Computes mean and 95% bootstrap confidence interval.
    Returns: (mean, lower_bound, upper_bound)
    """
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.array(values)
    mean_val = float(np.mean(arr))
    if len(arr) < 5:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstrap):
        sampled = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sampled))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, lower, upper


def compute_paired_bootstrap_test(
    values_a: List[float],
    values_b: List[float],
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Paired bootstrap test for per-query metric differences.
    Returns: (p_value, mean_difference)
    H0: mean difference = 0 (no significant difference)
    Uses centered bootstrap (shift differences by observed mean) for proper H0 testing.
    """
    if len(values_a) != len(values_b) or not values_a:
        return 1.0, 0.0

    diffs = np.array([a - b for a, b in zip(values_a, values_b)])
    observed_diff = float(np.mean(diffs))
    n = len(diffs)

    # Center differences under H0: mean difference = 0
    diffs_centered = diffs - observed_diff

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diffs_centered[idx])))

    # Two-sided p-value: proportion of bootstrap samples with |mean| >= |observed|
    p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(observed_diff)))
    return p_value, observed_diff


def compute_randomization_test(
    values_a: List[float],
    values_b: List[float],
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Randomization test (permutation test) for per-query metric differences.
    Returns: (p_value, mean_difference)
    H0: the two models have identical performance distributions
    """
    if len(values_a) != len(values_b) or not values_a:
        return 1.0, 0.0

    diffs = np.array([a - b for a, b in zip(values_a, values_b)])
    observed_diff = float(np.mean(diffs))
    n = len(diffs)

    rng = np.random.default_rng(seed)
    perm_diffs = []
    for _ in range(n_permutations):
        # Randomly flip signs (equivalent to swapping model assignments per query)
        signs = rng.choice([-1, 1], size=n)
        perm_diffs.append(float(np.mean(diffs * signs)))

    # Two-sided p-value: proportion of permuted means with |mean| >= |observed|
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))
    return p_value, observed_diff


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Holm-Bonferroni step-down procedure for multiple comparison correction.
    Returns: (rejected: List[bool], adjusted_p_values: List[float])
    """
    if not p_values:
        return [], []
    
    n = len(p_values)
    indexed_p = [(p, i) for i, p in enumerate(p_values)]
    indexed_p.sort(key=lambda x: x[0])  # Sort by p-value ascending
    
    rejected = [False] * n
    adjusted = [0.0] * n
    
    for rank, (p_val, orig_idx) in enumerate(indexed_p):
        # Holm-Bonferroni: compare p_(k) with alpha / (n - k + 1)
        threshold = alpha / (n - rank)
        adj_p = min(p_val * (n - rank), 1.0)
        adjusted[orig_idx] = adj_p
        if p_val <= threshold:
            rejected[orig_idx] = True
        else:
            # Once we fail to reject, all subsequent (larger p-values) are also not rejected
            break
    
    return rejected, adjusted

