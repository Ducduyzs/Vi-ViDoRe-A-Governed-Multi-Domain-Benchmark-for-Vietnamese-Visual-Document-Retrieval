import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import PathConfig, BenchmarkConfig, LLMConfig
from src.data.schema import PageMetadata, QueryItem, QrelItem, BenchmarkSplit, DomainType, QueryType
from src.data.query_sanitizer import QuerySanitizer
from src.data.query_generator import QueryGenerator
from src.data.deduplication import DatasetAuditor

def split_documents_anti_leakage(
    pages: List[PageMetadata],
    test_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits documents (by doc_id) across train, dev, and test sets.
    Guarantees no pages from the same PDF document appear in both train and test.
    """
    random.seed(seed)
    doc_ids = sorted(list({p.doc_id for p in pages}))
    random.shuffle(doc_ids)

    n = len(doc_ids)
    n_test = max(1, int(n * test_ratio))
    n_dev = max(1, int(n * dev_ratio))

    test_docs = set(doc_ids[:n_test])
    dev_docs = set(doc_ids[n_test : n_test + n_dev])
    train_docs = set(doc_ids[n_test + n_dev :])

    return list(train_docs), list(dev_docs), list(test_docs)

def generate_queries_for_pages(
    pages: List[PageMetadata],
    split_name: str,
    query_generator: QueryGenerator,
    max_workers: int = 4,
    sample_pages_ratio: float = 1.0,
    max_failures: int = 10,
) -> Tuple[List[QueryItem], List[QrelItem]]:
    """
    Generates queries for a subset of pages using LLM API / Heuristics.
    """
    valid_pages = [p for p in pages if len(p.native_text.strip()) >= 50]
    if sample_pages_ratio < 1.0:
        k = max(1, int(len(valid_pages) * sample_pages_ratio))
        valid_pages = random.sample(valid_pages, k)

    all_queries: List[QueryItem] = []
    all_qrels: List[QrelItem] = []
    failure_count = 0

    def process_single_page(p: PageMetadata) -> List[QueryItem]:
        return query_generator.generate_queries_for_page(
            domain=p.domain,
            page_text=p.native_text,
            page_num=p.page_num,
            doc_id=p.doc_id,
            target_page_id=p.page_id,
            query_id_prefix=f"q_{split_name}",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_page, p): p for p in valid_pages}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating queries ({split_name})"):
            try:
                items = future.result()
                for q_item in items:
                    all_queries.append(q_item)
                    for pid in q_item.target_page_ids:
                        all_qrels.append(QrelItem(query_id=q_item.query_id, page_id=pid, relevance=2))
            except Exception as e:
                failure_count += 1
                page = futures[future]
                print(f"[!] Query generation failed for page {page.page_id}: {e}")
                import traceback
                traceback.print_exc()
                if failure_count >= max_failures:
                    print(f"[!] Too many failures ({failure_count} >= {max_failures}), stopping...")
                    raise SystemExit(1)

    if failure_count > 0:
        print(f"[!] Query generation completed with {failure_count} failure(s)")
    
    return all_queries, all_qrels

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate benchmark queries and create leak-free splits.")
    parser.add_argument("--test_ratio", type=float, default=0.6, help="Test split document ratio")
    parser.add_argument("--dev_ratio", type=float, default=0.2, help="Dev split document ratio")
    parser.add_argument("--sample_ratio", type=float, default=0.5, help="Ratio of pages to generate queries for")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent LLM API threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--max_failures", type=int, default=10, help="Max query generation failures before aborting")
    args = parser.parse_args()

    paths = PathConfig()
    paths.make_dirs()

    meta_path = paths.processed_dir / "all_pages_metadata.jsonl"
    if not meta_path.exists():
        print(f"[!] Metadata file not found at {meta_path}. Please run scripts/01_process_pdfs.py first.")
        return

    pages: List[PageMetadata] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(PageMetadata.from_dict(json.loads(line)))

    print(f"[*] Loaded {len(pages)} pages from {meta_path}")

    # Load LLM Config from config.local.json
    llm_config = LLMConfig.load_from_file(paths.local_config_file)
    print(f"[*] Loaded LLM Provider: {llm_config.provider} (OpenAI: {'Yes' if llm_config.openai_api_key else 'No'}, Gemini: {'Yes' if llm_config.gemini_api_key else 'No'})")

    sanitizer = QuerySanitizer()
    generator = QueryGenerator(sanitizer=sanitizer, llm_config=llm_config)

    train_docs, dev_docs, test_docs = split_documents_anti_leakage(
        pages, test_ratio=args.test_ratio, dev_ratio=args.dev_ratio, seed=args.seed
    )

    print(f"[*] Anti-leakage Document Split:")
    print(f"    - Train documents: {len(train_docs)}")
    print(f"    - Dev documents:   {len(dev_docs)}")
    print(f"    - Test documents:  {len(test_docs)}")

    train_pages = [p for p in pages if p.doc_id in train_docs]
    dev_pages = [p for p in pages if p.doc_id in dev_docs]
    test_pages = [p for p in pages if p.doc_id in test_docs]

    # Audit leakage between train and test
    auditor = DatasetAuditor()
    leakages = auditor.audit_train_test_leakage(
        [p.to_dict() for p in train_pages], [p.to_dict() for p in test_pages]
    )
    if leakages:
        print(f"[!] WARNING: Found {len(leakages)} potential leakages between train and test:")
        for l in leakages[:5]:
            print(f"    - {l}")
    else:
        print("[+] Audit Passed: Zero exact or near-duplicate leakage between train and test!")

    # Generate queries for each split
    test_queries, test_qrels = generate_queries_for_pages(
        test_pages, "test", generator, max_workers=args.workers, sample_pages_ratio=args.sample_ratio, max_failures=args.max_failures
    )
    dev_queries, dev_qrels = generate_queries_for_pages(
        dev_pages, "dev", generator, max_workers=args.workers, sample_pages_ratio=args.sample_ratio, max_failures=args.max_failures
    )
    train_queries, train_qrels = generate_queries_for_pages(
        train_pages, "train", generator, max_workers=args.workers, sample_pages_ratio=args.sample_ratio, max_failures=args.max_failures
    )

    test_split = BenchmarkSplit(
        name="test",
        queries=test_queries,
        corpus_page_ids=[p.page_id for p in test_pages],
        qrels=test_qrels,
        doc_sources=test_docs,
    )
    dev_split = BenchmarkSplit(
        name="dev",
        queries=dev_queries,
        corpus_page_ids=[p.page_id for p in dev_pages],
        qrels=dev_qrels,
        doc_sources=dev_docs,
    )
    train_split = BenchmarkSplit(
        name="train",
        queries=train_queries,
        corpus_page_ids=[p.page_id for p in train_pages],
        qrels=train_qrels,
        doc_sources=train_docs,
    )

    test_split.save(paths.benchmark_dir / "test")
    dev_split.save(paths.benchmark_dir / "dev")
    train_split.save(paths.benchmark_dir / "train")

    print(f"\n[+] Benchmark splits saved successfully to: {paths.benchmark_dir}")
    print(f"    - Test queries: {len(test_split.queries):<5} | Corpus: {len(test_split.corpus_page_ids)} pages")
    print(f"    - Dev queries:  {len(dev_split.queries):<5} | Corpus: {len(dev_split.corpus_page_ids)} pages")
    print(f"    - Train queries:{len(train_split.queries):<5} | Corpus: {len(train_split.corpus_page_ids)} pages")

if __name__ == "__main__":
    main()
