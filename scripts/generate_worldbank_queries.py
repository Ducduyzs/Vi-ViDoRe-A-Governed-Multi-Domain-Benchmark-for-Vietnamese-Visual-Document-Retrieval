"""Generate queries for World Bank documents using heuristic fallback."""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from src.data.query_generator import QueryGenerator, VIETNAMESE_PROMPT_TEMPLATES
from src.data.schema import DomainType, QueryType, QueryItem
from src.data.query_sanitizer import QuerySanitizer


def load_pages_metadata(path: Path):
    pages = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def generate_queries_for_doc(pages, doc_id, domain, query_id_prefix):
    """Generate queries for a document using heuristic fallback."""
    generator = QueryGenerator()
    all_queries = []
    
    for page in pages:
        if page["doc_id"] != doc_id:
            continue
        
        native_text = page.get("native_text", "")
        if not native_text or len(native_text.strip()) < 30:
            continue
        
        page_num = page["page_num"]
        page_id = page["page_id"]
        
        queries = generator.generate_queries_for_page(
            domain=domain,
            page_text=native_text,
            page_num=page_num,
            doc_id=doc_id,
            target_page_id=page_id,
            query_id_prefix=query_id_prefix,
        )
        
        for q in queries:
            q.metadata.update({
                "governance_status": "pending_human_validation",
                "candidate_label_only": True,
                "assigned_split": "test",
            })
            all_queries.append(q.to_dict())
    
    return all_queries


def main():
    pages_path = Path("data/benchmark_governed_v0_1/test/pages_metadata.jsonl")
    output_path = Path("data/benchmark_governed_v0_1/test/queries_candidates.jsonl")
    
    pages = load_pages_metadata(pages_path)
    
    existing_queries = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_queries.append(json.loads(line))
    
    existing_ids = {q["query_id"] for q in existing_queries}
    
    financial_queries = generate_queries_for_doc(
        pages, "worldbank_innovation_east_asia_overview_vi", 
        DomainType.FINANCIAL, "q_test_wb_fin"
    )
    
    healthcare_queries = generate_queries_for_doc(
        pages, "worldbank_bao_phu_bao_hiem_y_te_viet_nam_vi", 
        DomainType.HEALTHCARE, "q_test_wb_hc"
    )
    
    new_queries = [q for q in financial_queries + healthcare_queries 
                   if q["query_id"] not in existing_ids]
    
    all_queries = existing_queries + new_queries
    
    with output_path.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")
    
    print(f"Added {len(new_queries)} new queries ({len(financial_queries)} financial, {len(healthcare_queries)} healthcare)")
    print(f"Total queries in test: {len(all_queries)}")


if __name__ == "__main__":
    main()