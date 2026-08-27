#!/usr/bin/env python3
"""
Pool candidates from multiple retrievers for human annotation.
Combines BM25, Dense Vietnamese (BGE-M3), ColPali, ColQwen.
"""

import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PathConfig
from src.data.schema import PageMetadata, QueryItem, QrelItem, BenchmarkSplit
from src.models.text_baselines import BM25Retriever, DenseBiEncoderRetriever
from src.models.visual_retriever import ColPaliVisualRetriever

def load_split(split_dir: Path) -> BenchmarkSplit:
    queries = []
    with open(split_dir / "queries_candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(QueryItem.from_dict(json.loads(line)))
    
    # Load corpus pages
    with open(split_dir / "corpus_pages.json", "r", encoding="utf-8") as f:
        corpus_page_ids = json.load(f)
    
    # Load qrels if exist (for evaluation)
    qrels = []
    qrels_path = split_dir / "qrels.tsv"
    if qrels_path.exists():
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qrels.append(QrelItem.from_tsv_row(line))
    
    return BenchmarkSplit(
        name=split_dir.name,
        queries=queries,
        corpus_page_ids=corpus_page_ids,
        qrels=qrels,
        doc_sources=[],
    )

def load_metadata(meta_path: Path):
    pages_meta_map = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = PageMetadata.from_dict(json.loads(line))
                    pages_meta_map[p.page_id] = p
    return pages_meta_map

def retrieve_bm25(split, pages_meta_map, top_k=50):
    print("[*] Running BM25 retrieval...")
    bm25 = BM25Retriever()
    corpus_texts = [pages_meta_map[pid].native_text for pid in split.corpus_page_ids if pid in pages_meta_map]
    bm25.index_corpus(split.corpus_page_ids, corpus_texts)
    
    query_texts = [q.query_text for q in split.queries]
    query_ids = [q.query_id for q in split.queries]
    results = bm25.retrieve(query_texts, query_ids=query_ids, top_k=top_k)
    return results

def retrieve_dense(split, pages_meta_map, model_name="bkai-foundation-models/vietnamese-bi-encoder", top_k=50):
    print(f"[*] Running Dense Bi-Encoder ({model_name})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi_encoder = DenseBiEncoderRetriever(model_name_or_path=model_name, device=device)
    corpus_texts = [pages_meta_map[pid].native_text for pid in split.corpus_page_ids if pid in pages_meta_map]
    bi_encoder.index_corpus(split.corpus_page_ids, corpus_texts)
    
    query_texts = [q.query_text for q in split.queries]
    query_ids = [q.query_id for q in split.queries]
    results = bi_encoder.retrieve(query_texts, query_ids=query_ids, top_k=top_k)
    return results

def retrieve_visual(split, pages_meta_map, model_name="vidore/colpali-v1.2", revision=None, top_k=50):
    print(f"[*] Running Visual Retriever ({model_name})...")
    retriever = ColPaliVisualRetriever(
        model_name_or_path=model_name,
        revision=revision,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    image_paths = [pages_meta_map[pid].image_path for pid in split.corpus_page_ids if pid in pages_meta_map]
    retriever.index_corpus_from_images(split.corpus_page_ids, image_paths, batch_size=1)
    
    query_texts = [q.query_text for q in split.queries]
    query_ids = [q.query_id for q in split.queries]
    results = retriever.retrieve(query_texts, query_ids=query_ids, top_k=top_k, batch_size=2)
    return results

def merge_candidates(results_dict, top_k_per_retriever=20):
    """
    Merge candidates from multiple retrievers using reciprocal rank fusion (RRF).
    results_dict: {retriever_name: {query_id: [(page_id, score), ...]}}
    """
    from collections import defaultdict
    
    k = 60  # RRF constant
    merged = defaultdict(dict)
    
    for retriever_name, results in results_dict.items():
        for query_id, ranked_list in results.items():
            for rank, (page_id, score) in enumerate(ranked_list, 1):
                rrf_score = 1.0 / (k + rank)
                if page_id not in merged[query_id] or rrf_score > merged[query_id][page_id]:
                    merged[query_id][page_id] = rrf_score
    
    # Convert to ranked lists
    final_results = {}
    for query_id, page_scores in merged.items():
        ranked = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        final_results[query_id] = ranked[:top_k_per_retriever]
    
    return final_results

def save_pooled_candidates(pooled_results, output_path, existing_candidate_pages=None):
    """Save pooled candidates as annotation template."""
    import csv
    
    # If existing candidate pages provided, mark them
    existing = set()
    if existing_candidate_pages:
        for qid, pages in existing_candidate_pages.items():
            for pid in pages:
                existing.add((qid, pid))
    
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "page_id", "annotator_id", "relevance",
            "query_status", "evidence_note", "judged_at",
            "guideline_version", "adjudicated_relevance", "candidate_source_page"
        ], delimiter="\t")
        writer.writeheader()
        
        for query_id, ranked_list in pooled_results.items():
            for page_id, score in ranked_list:
                writer.writerow({
                    "query_id": query_id,
                    "page_id": page_id,
                    "annotator_id": "",
                    "relevance": "",
                    "query_status": "PENDING",
                    "evidence_note": "",
                    "judged_at": "",
                    "guideline_version": "1.0",
                    "adjudicated_relevance": "",
                    "candidate_source_page": "true" if (query_id, page_id) in existing else "false",
                })

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pool candidates from multiple retrievers")
    parser.add_argument("--split", default="test", help="Split to pool")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K per retriever")
    parser.add_argument("--final_top_k", type=int, default=30, help="Final top-K after merging")
    parser.add_argument("--dense_model", default="bkai-foundation-models/vietnamese-bi-encoder", help="Dense model")
    parser.add_argument("--visual_model", default="vidore/colpali-v1.2", help="Visual model")
    parser.add_argument("--visual_revision", default=None, help="Visual model revision")
    parser.add_argument("--skip_bm25", action="store_true", help="Skip BM25")
    parser.add_argument("--skip_dense", action="store_true", help="Skip Dense")
    parser.add_argument("--skip_visual", action="store_true", help="Skip Visual")
    args = parser.parse_args()
    
    paths = PathConfig()
    split_dir = paths.benchmark_dir / args.split
    if not split_dir.exists():
        # Try governed candidate
        split_dir = paths.data_dir / "benchmark_governed_v0_1" / args.split
    
    print(f"[*] Loading split from {split_dir}")
    split = load_split(split_dir)
    print(f"    {len(split.queries)} queries, {len(split.corpus_page_ids)} corpus pages")
    
    meta_path = paths.processed_dir / "all_pages_metadata.jsonl"
    pages_meta_map = load_metadata(meta_path)
    pages_meta_list = [pages_meta_map[pid] for pid in split.corpus_page_ids if pid in pages_meta_map]
    
    # Load existing candidate pages
    existing_candidates = {}
    candidate_queries_path = split_dir / "queries_candidates.jsonl"
    if candidate_queries_path.exists():
        with open(candidate_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    existing_candidates[q["query_id"]] = q.get("target_page_ids", [])
    
    all_results = {}
    
    # Run retrievers
    if not args.skip_bm25:
        try:
            all_results["bm25"] = retrieve_bm25(split, pages_meta_map, top_k=args.top_k)
        except Exception as e:
            print(f"[!] BM25 failed: {e}")
    
    if not args.skip_dense:
        try:
            all_results["dense_vietnamese"] = retrieve_dense(split, pages_meta_map, args.dense_model, top_k=args.top_k)
        except Exception as e:
            print(f"[!] Dense failed: {e}")
    
    if not args.skip_visual:
        try:
            all_results["colpali"] = retrieve_visual(split, pages_meta_map, args.visual_model, args.visual_revision, top_k=args.top_k)
        except Exception as e:
            print(f"[!] Visual failed: {e}")
    
    # Merge
    print(f"\n[*] Merging candidates from {len(all_results)} retrievers using RRF...")
    pooled = merge_candidates(all_results, top_k_per_retriever=args.final_top_k)
    
    # Stats
    total_pairs = sum(len(v) for v in pooled.values())
    avg_per_query = total_pairs / len(pooled) if pooled else 0
    print(f"    Total query-page pairs: {total_pairs}")
    print(f"    Avg per query: {avg_per_query:.1f}")
    
    # Save
    output_dir = paths.data_dir / "benchmark_governed_v0_1" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_tsv = output_dir / f"annotations_pooled_top{args.final_top_k}_template.tsv"
    save_pooled_candidates(pooled, output_tsv, existing_candidates)
    print(f"[+] Pooled template saved to {output_tsv}")
    
    # Also save per-retriever results for analysis
    output_json = output_dir / f"pooled_candidates_top{args.final_top_k}.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "num_retrievers": len(all_results),
            "retrievers": list(all_results.keys()),
            "total_pairs": total_pairs,
            "pooled_results": {qid: [(pid, float(score)) for pid, score in ranked] for qid, ranked in pooled.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"[+] Detailed results saved to {output_json}")

if __name__ == "__main__":
    main()