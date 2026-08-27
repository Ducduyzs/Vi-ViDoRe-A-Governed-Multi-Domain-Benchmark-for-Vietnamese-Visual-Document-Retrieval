from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

from src.data.schema import QueryItem, PageMetadata, QrelItem, BenchmarkSplit
from src.models.base import BaseRetriever
from src.evaluation.metrics import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_mrr_at_k,
    compute_bootstrap_ci,
    compute_paired_bootstrap_test,
    compute_randomization_test,
    holm_bonferroni_correction,
)

class ViViDoReEvaluator:
    """
    Evaluation harness for the Vi-ViDoRe benchmark.
    Computes overall metrics, confidence intervals, and sliced breakdowns:
    - by Domain (Legal, Financial, Healthcare, Education, Infographic)
    - by Source Type (Born-digital vs Scanned)
    - by Page Type (Text, Table, Chart, Mixed)
    - by Query Generation (Human-written vs LLM-assisted)
    """
    def __init__(
        self,
        split: BenchmarkSplit,
        pages_metadata: Optional[List[PageMetadata]] = None,
        top_k_list: Optional[List[int]] = None,
        bootstrap_seed: Optional[int] = None,
    ):
        self.split = split
        self.queries = split.queries
        self.query_dict = {q.query_id: q for q in split.queries}
        self.top_k_list = top_k_list or [1, 5, 10, 20]
        self.bootstrap_seed = bootstrap_seed

        # Build qrels mapping: query_id -> {page_id: relevance}
        self.qrels_map: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in split.qrels:
            self.qrels_map[qrel.query_id][qrel.page_id] = qrel.relevance

        # Pages metadata dictionary
        self.page_dict: Dict[str, PageMetadata] = {}
        if pages_metadata:
            self.page_dict = {p.page_id: p for p in pages_metadata}

    def evaluate_retrieval_results(
        self,
        retrieval_results: Dict[str, List[Tuple[str, float]]],
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Evaluates pre-computed retrieval results.
        retrieval_results: dict mapping query_id -> list of (page_id, score)
        """
        all_metrics = {f"ndcg@{k}": [] for k in self.top_k_list}
        all_metrics.update({f"recall@{k}": [] for k in self.top_k_list})
        all_metrics["mrr@10"] = []

        # Per-query metrics for significance testing
        per_query_metrics: Dict[str, Dict[str, float]] = {}

        # Slice containers
        domain_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        domain_counts: Dict[str, int] = defaultdict(int)
        source_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        source_counts: Dict[str, int] = defaultdict(int)
        query_type_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        query_type_counts: Dict[str, int] = defaultdict(int)
        creation_source_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        creation_source_counts: Dict[str, int] = defaultdict(int)

        for q in self.queries:
            q_id = q.query_id
            gold_qrels = self.qrels_map.get(q_id, {})
            retrieved_pairs = retrieval_results.get(q_id, [])
            retrieved_pids = [pid for pid, _ in retrieved_pairs]

            # Domain & Metadata info
            domain = q.domain.value
            q_type = q.query_type.value
            source = q.source

            # Source type: determine from all positive pages (majority vote)
            source_type = "unknown"
            if q.target_page_ids:
                source_type_votes = []
                for pid in q.target_page_ids:
                    meta = self.page_dict.get(pid)
                    if meta and meta.source_type:
                        source_type_votes.append(meta.source_type.value)
                if source_type_votes:
                    source_type = max(set(source_type_votes), key=source_type_votes.count)

            # Compute metrics for this query
            q_results = {}
            for k in self.top_k_list:
                ndcg = compute_ndcg_at_k(retrieved_pids, gold_qrels, k=k)
                rec = compute_recall_at_k(retrieved_pids, gold_qrels, k=k)
                all_metrics[f"ndcg@{k}"].append(ndcg)
                all_metrics[f"recall@{k}"].append(rec)
                q_results[f"ndcg@{k}"] = ndcg
                q_results[f"recall@{k}"] = rec

            mrr = compute_mrr_at_k(retrieved_pids, gold_qrels, k=10)
            all_metrics["mrr@10"].append(mrr)
            q_results["mrr@10"] = mrr

            # Store per-query metrics
            per_query_metrics[q_id] = q_results

            # Populate slice breakdowns
            for k_metric, val in q_results.items():
                domain_metrics[domain][k_metric].append(val)
                source_metrics[source_type][k_metric].append(val)
                query_type_metrics[q_type][k_metric].append(val)
                creation_source_metrics[source][k_metric].append(val)
            domain_counts[domain] += 1
            source_counts[source_type] += 1
            query_type_counts[q_type] += 1
            creation_source_counts[source] += 1

        # Aggregate overall summary with 95% Confidence Intervals
        overall_summary = {}
        for metric_name, values in all_metrics.items():
            mean_val, lower, upper = compute_bootstrap_ci(values, seed=self.bootstrap_seed)
            overall_summary[metric_name] = {
                "mean": round(mean_val, 4),
                "ci_95": [round(lower, 4), round(upper, 4)],
            }

        # Aggregate Macro-Domain Average
        domain_summary = {}
        domain_ndcg5_list = []
        for dom, metrics in domain_metrics.items():
            domain_summary[dom] = {
                m: round(float(np.mean(vals)), 4) for m, vals in metrics.items()
            }
            if "ndcg@5" in domain_summary[dom]:
                domain_ndcg5_list.append(domain_summary[dom]["ndcg@5"])

        macro_domain_ndcg5 = round(float(np.mean(domain_ndcg5_list)), 4) if domain_ndcg5_list else 0.0

        def _build_slice_summary(metrics_dict: Dict[str, Dict[str, List[float]]], counts_dict: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
            """Build slice summary with metrics and counts."""
            result = {}
            for slice_name, metrics in metrics_dict.items():
                result[slice_name] = {
                    m: round(float(np.mean(vals)), 4) for m, vals in metrics.items()
                }
                result[slice_name]["count"] = counts_dict.get(slice_name, 0)
            return result

        return {
            "model_name": model_name,
            "num_queries": len(self.queries),
            "macro_domain_ndcg@5": macro_domain_ndcg5,
            "overall": overall_summary,
            "by_domain": _build_slice_summary(domain_metrics, domain_counts),
            "by_source_type": _build_slice_summary(source_metrics, source_counts),
            "by_query_type": _build_slice_summary(query_type_metrics, query_type_counts),
            "by_creation_source": _build_slice_summary(creation_source_metrics, creation_source_counts),
            "per_query": per_query_metrics,
        }

    def significance_test(
        self,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        metric: str = "ndcg@5",
        method: str = "paired_bootstrap",
    ) -> Dict[str, Any]:
        """
        Perform significance test between two models' results.
        Args:
            results_a: Results from evaluate_retrieval_results for model A
            results_b: Results from evaluate_retrieval_results for model B
            metric: Metric to test (e.g., "ndcg@5", "mrr@10")
            method: "paired_bootstrap" or "randomization"
        Returns:
            Dict with p_value, mean_difference, and interpretation
        """
        per_query_a = results_a.get("per_query", {})
        per_query_b = results_b.get("per_query", {})

        common_qids = set(per_query_a.keys()) & set(per_query_b.keys())
        if not common_qids:
            return {"error": "No common queries between results"}

        values_a = [per_query_a[qid].get(metric, 0.0) for qid in common_qids]
        values_b = [per_query_b[qid].get(metric, 0.0) for qid in common_qids]

        if method == "paired_bootstrap":
            p_value, mean_diff = compute_paired_bootstrap_test(values_a, values_b, seed=self.bootstrap_seed)
        elif method == "randomization":
            p_value, mean_diff = compute_randomization_test(values_a, values_b, seed=self.bootstrap_seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            "metric": metric,
            "method": method,
            "model_a": results_a.get("model_name", "A"),
            "model_b": results_b.get("model_name", "B"),
            "p_value": round(p_value, 6),
            "mean_difference": round(mean_diff, 6),
            "significant_at_05": p_value < 0.05,
            "significant_at_01": p_value < 0.01,
            "n_queries": len(common_qids),
            "interpretation": f"{'Significant' if p_value < 0.05 else 'Not significant'} difference (p={p_value:.4f})"
        }

    def pairwise_significance_matrix(
        self,
        all_results: List[Dict[str, Any]],
        metric: str = "ndcg@5",
        method: str = "paired_bootstrap",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run all pairwise significance tests between models and apply Holm-Bonferroni correction.
        Args:
            all_results: List of results from evaluate_retrieval_results for all models
            metric: Metric to test
            method: "paired_bootstrap" or "randomization"
            alpha: Significance level for correction
        Returns:
            Dict with pairwise results and Holm-Bonferroni corrected results
        """
        model_names = [r.get("model_name", f"Model_{i}") for i, r in enumerate(all_results)]
        n_models = len(all_results)
        
        pairwise_results = []
        p_values = []
        comparisons = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                comp_key = f"{model_names[i]} vs {model_names[j]}"
                comparisons.append(comp_key)
                result = self.significance_test(all_results[i], all_results[j], metric, method)
                pairwise_results.append({
                    "comparison": comp_key,
                    "model_a": model_names[i],
                    "model_b": model_names[j],
                    **result
                })
                p_values.append(result.get("p_value", 1.0))
        
        # Apply Holm-Bonferroni correction
        rejected, adjusted_p = holm_bonferroni_correction(p_values, alpha)
        
        # Combine results
        corrected_results = []
        for idx, (pair, reject, adj_p) in enumerate(zip(pairwise_results, rejected, adjusted_p)):
            corrected_results.append({
                **pair,
                "p_value_adjusted": round(adj_p, 6),
                "significant_after_correction": reject,
                "correction_method": "holm_bonferroni",
                "alpha": alpha,
            })
        
        return {
            "metric": metric,
            "method": method,
            "alpha": alpha,
            "correction": "holm_bonferroni",
            "pairwise": corrected_results,
            "summary": {
                "total_comparisons": len(comparisons),
                "significant_before_correction": sum(1 for p in p_values if p < alpha),
                "significant_after_correction": sum(rejected),
            }
        }

