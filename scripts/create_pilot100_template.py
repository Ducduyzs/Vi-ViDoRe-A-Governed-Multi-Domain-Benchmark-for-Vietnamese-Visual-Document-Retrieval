#!/usr/bin/env python3
"""
Create annotation template TSV for pilot 100 queries.
Samples from test split candidate queries, ensures domain balance.
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_QUERIES = ROOT / "data" / "benchmark_governed_v0_1" / "test" / "queries_candidates.jsonl"
OUTPUT_DIR = ROOT / "data" / "benchmark_governed_v0_1" / "test"
OUTPUT_TSV = OUTPUT_DIR / "annotations_pilot100_template.tsv"
OUTPUT_MANIFEST = OUTPUT_DIR / "pilot100_manifest.json"

# Target: 100 queries, balanced across domains
TARGET_PER_DOMAIN = {
    "education": 25,
    "legal": 25,
    "financial": 25,
    "healthcare": 25,
}

random.seed(42)

def main():
    # Load candidate queries
    queries_by_domain = defaultdict(list)
    with CANDIDATE_QUERIES.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                domain = q.get("domain", "unknown")
                if domain in TARGET_PER_DOMAIN:
                    queries_by_domain[domain].append(q)
    
    print("Available candidate queries per domain:")
    for domain, qs in queries_by_domain.items():
        print(f"  {domain}: {len(qs)}")
    
    # Sample per domain
    selected = []
    for domain, target in TARGET_PER_DOMAIN.items():
        available = queries_by_domain.get(domain, [])
        if len(available) < target:
            print(f"WARNING: {domain} only has {len(available)} queries, need {target}")
            sampled = available
        else:
            sampled = random.sample(available, target)
        selected.extend(sampled)
        print(f"  Selected {len(sampled)} from {domain}")
    
    print(f"\nTotal selected: {len(selected)}")
    
    # Write TSV template
    columns = [
        "query_id",
        "page_id",
        "annotator_id",
        "relevance",
        "query_status",
        "evidence_note",
        "judged_at",
        "guideline_version",
        "adjudicated_relevance",
        "candidate_source_page",
    ]
    
    with OUTPUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for query in selected:
            for page_id in query.get("target_page_ids", []):
                writer.writerow({
                    "query_id": query["query_id"],
                    "page_id": page_id,
                    "annotator_id": "",
                    "relevance": "",
                    "query_status": "PENDING",
                    "evidence_note": "",
                    "judged_at": "",
                    "guideline_version": "1.0",
                    "adjudicated_relevance": "",
                    "candidate_source_page": "true",
                })
    
    print(f"\n[+] Template written to {OUTPUT_TSV}")
    
    # Write manifest
    manifest = {
        "pilot_name": "pilot100",
        "created_at": "2026-08-27",
        "guideline_version": "1.0",
        "target_per_domain": TARGET_PER_DOMAIN,
        "actual_per_domain": {d: len([q for q in selected if q.get("domain") == d]) for d in TARGET_PER_DOMAIN},
        "total_queries": len(selected),
        "total_pairs": sum(len(q.get("target_page_ids", [])) for q in selected),
        "query_ids": [q["query_id"] for q in selected],
        "split": "test",
        "source": "queries_candidates.jsonl",
        "sampling_seed": 42,
    }
    
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Manifest written to {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()