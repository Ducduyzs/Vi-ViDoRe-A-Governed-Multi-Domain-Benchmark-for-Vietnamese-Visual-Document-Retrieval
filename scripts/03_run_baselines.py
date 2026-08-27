import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import traceback
from typing import List, Dict, Any
import torch

from src.config import PathConfig, BenchmarkConfig, ModelConfig
from src.data.schema import PageMetadata, QueryItem, QrelItem, BenchmarkSplit
from src.models.text_baselines import BM25Retriever, DenseBiEncoderRetriever
from src.evaluation.evaluator import ViViDoReEvaluator
from src.evaluation.report_generator import save_evaluation_report


def load_split(split_dir: Path) -> BenchmarkSplit:
    queries: List[QueryItem] = []
    with open(split_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(QueryItem.from_dict(json.loads(line)))

    qrels: List[QrelItem] = []
    with open(split_dir / "qrels.tsv", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qrels.append(QrelItem.from_tsv_row(line))

    with open(split_dir / "corpus_pages.json", "r", encoding="utf-8") as f:
        corpus_page_ids = json.load(f)

    return BenchmarkSplit(
        name=split_dir.name,
        queries=queries,
        corpus_page_ids=corpus_page_ids,
        qrels=qrels,
        doc_sources=[],
    )


def run_baseline(name: str, baseline_fn, mandatory: bool = True) -> Dict[str, Any]:
    """Run a baseline evaluation. Fail-fast on any error for mandatory baselines."""
    try:
        return {"success": True, "metrics": baseline_fn(), "error": None}
    except Exception as e:
        error_msg = f"[!] Error running {name}: {e}\n{traceback.format_exc()}"
        if mandatory:
            print(error_msg)
            raise SystemExit(1)
        else:
            print(f"{error_msg} (continuing...)")
            return {"success": False, "metrics": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Step 3: Run baseline evaluation on Vi-ViDoRe benchmark.")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate: test or dev")
    parser.add_argument("--biencoder_model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Dense Bi-Encoder model name")
    parser.add_argument("--skip_biencoder", action="store_true", help="Skip Dense Bi-Encoder evaluation")
    parser.add_argument("--run_visual", action=argparse.BooleanOptionalAction, default=True, help="Evaluate ColPali visual model (Zero-Shot)")
    parser.add_argument("--visual_model", type=str, default="vidore/colpali-v1.2", help="ColPali model checkpoint")
    parser.add_argument("--visual_revision", type=str, default=None, help="Hugging Face model revision (commit hash) for visual model")
    args = parser.parse_args()

    paths = PathConfig()
    split_dir = paths.benchmark_dir / args.split
    if not split_dir.exists():
        print(f"[!] Split directory not found at {split_dir}. Please run scripts/02_generate_benchmark.py first.")
        return

    split = load_split(split_dir)
    print(f"[*] Loaded {args.split} split: {len(split.queries)} queries, {len(split.corpus_page_ids)} corpus pages.")

    # Load page metadata
    meta_path = paths.processed_dir / "all_pages_metadata.jsonl"
    pages_meta_map = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = PageMetadata.from_dict(json.loads(line))
                    pages_meta_map[p.page_id] = p

    pages_meta_list = [pages_meta_map[pid] for pid in split.corpus_page_ids if pid in pages_meta_map]
    bench_config = BenchmarkConfig()
    evaluator = ViViDoReEvaluator(split, pages_metadata=pages_meta_list, bootstrap_seed=bench_config.ci_bootstrap_seed)

    query_texts = [q.query_text for q in split.queries]
    query_ids = [q.query_id for q in split.queries]
    corpus_texts = [pages_meta_map[pid].native_text for pid in split.corpus_page_ids if pid in pages_meta_map]

    all_results = []
    all_per_query_results = {}

    def run_and_save(name: str, baseline_fn, model_key: str):
        result = run_baseline(name, baseline_fn, mandatory=True)
        if not result["success"]:
            return
        metrics = result["metrics"]
        all_results.append(metrics)
        all_per_query_results[model_key] = metrics.get("per_query", {})
        print(f"    -> Macro nDCG@5: {metrics['macro_domain_ndcg@5']:.4f} | Overall nDCG@5: {metrics['overall']['ndcg@5']['mean']:.4f} | MRR@10: {metrics['overall']['mrr@10']['mean']:.4f}")

    # 1. Evaluate BM25 Baseline (MANDATORY)
    print("\n========================================================")
    print("[+] Evaluating Baseline 1: Native Text + BM25 (Lexical)...")
    print("========================================================")
    
    def run_bm25():
        bm25 = BM25Retriever()
        bm25.index_corpus(split.corpus_page_ids, corpus_texts)
        bm25_res = bm25.retrieve(query_texts, query_ids=query_ids, top_k=20)
        return evaluator.evaluate_retrieval_results(bm25_res, model_name="Native Text + BM25")
    
    run_and_save("Native Text + BM25", run_bm25, "bm25")

    # 2. Evaluate Dense Bi-Encoder Baseline (Optional but fail if not skipped)
    if not args.skip_biencoder:
        print("\n========================================================")
        print(f"[+] Evaluating Baseline 2: Dense Bi-Encoder ({args.biencoder_model})...")
        print("========================================================")
        
        def run_biencoder():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bi_encoder = DenseBiEncoderRetriever(model_name_or_path=args.biencoder_model, device=device)
            bi_encoder.index_corpus(split.corpus_page_ids, corpus_texts)
            bi_res = bi_encoder.retrieve(query_texts, query_ids=query_ids, top_k=20)
            return evaluator.evaluate_retrieval_results(bi_res, model_name=f"Dense Bi-Encoder ({Path(args.biencoder_model).name})")
        
        run_and_save(f"Dense Bi-Encoder ({args.biencoder_model})", run_biencoder, "biencoder")

    # 3. Evaluate Visual Late Interaction (ColPali Zero-Shot) (Optional but fail if explicitly requested)
    if args.run_visual:
        print("\n========================================================")
        print(f"[+] Evaluating Baseline 3: ColPali Zero-Shot Visual Retriever ({args.visual_model})...")
        print("========================================================")
        
        def run_visual():
            from src.models.visual_retriever import ColPaliVisualRetriever
            retriever = ColPaliVisualRetriever(
                model_name_or_path=args.visual_model,
                revision=args.visual_revision,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            image_paths = [pages_meta_map[pid].image_path for pid in split.corpus_page_ids]
            print(f"[*] Encoding {len(image_paths)} document images into multi-vector representations...")
            retriever.index_corpus_from_images(split.corpus_page_ids, image_paths, batch_size=1)
            print(f"[*] Encoding {len(query_texts)} queries and performing MaxSim retrieval...")
            vis_res = retriever.retrieve(query_texts, query_ids=query_ids, top_k=20, batch_size=2)
            return evaluator.evaluate_retrieval_results(vis_res, model_name="ColPali Zero-Shot")
        
        run_and_save(f"ColPali Zero-Shot ({args.visual_model})", run_visual, "colpali")

    # Save per-query artifact manifest
    artifact_dir = paths.results_dir / f"artifact_{args.split}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-query rankings and scores
    per_query_artifact = {
        "metadata": {
            "split": args.split,
            "biencoder_model": args.biencoder_model if not args.skip_biencoder else None,
            "visual_model": args.visual_model if args.run_visual else None,
            "visual_revision": args.visual_revision if args.run_visual else None,
            "bootstrap_seed": bench_config.ci_bootstrap_seed,
            "num_queries": len(split.queries),
            "corpus_pages": len(split.corpus_page_ids),
            "timestamp": datetime.now().isoformat(),
        },
        "per_query": all_per_query_results,
    }
    
    with open(artifact_dir / "per_query_rankings.json", "w", encoding="utf-8") as f:
        json.dump(per_query_artifact, f, ensure_ascii=False, indent=2)
    
    # Save run config
    run_config = {
        "split": args.split,
        "biencoder_model": args.biencoder_model if not args.skip_biencoder else None,
        "visual_model": args.visual_model if args.run_visual else None,
        "visual_revision": args.visual_revision if args.run_visual else None,
        "bootstrap_seed": bench_config.ci_bootstrap_seed,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    with open(artifact_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # Save and output reports
    run_metadata = {
        "split": args.split,
        "biencoder_model": args.biencoder_model if not args.skip_biencoder else None,
        "visual_model": args.visual_model if args.run_visual else None,
        "visual_revision": args.visual_revision if args.run_visual else None,
        "bootstrap_seed": bench_config.ci_bootstrap_seed,
    }
    save_evaluation_report(all_results, paths.results_dir, report_name=f"benchmark_{args.split}_results", run_metadata=run_metadata)
    print(f"\n[+] Full reports successfully updated in {paths.results_dir}:")
    print(f"    - Markdown: {paths.results_dir / f'benchmark_{args.split}_results.md'}")
    print(f"    - LaTeX:    {paths.results_dir / f'benchmark_{args.split}_results.tex'}")
    print(f"    - JSON:     {paths.results_dir / f'benchmark_{args.split}_results.json'}")
    print(f"    - Per-query artifact: {artifact_dir / 'per_query_rankings.json'}")
    print(f"    - Run config: {artifact_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()