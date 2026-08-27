#!/usr/bin/env python3
"""
Human-written query template and domain balancing script.
Helps create ≥40% human-written queries, balance domains.
"""

import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_QUERIES = ROOT / "data" / "benchmark_governed_v0_1" / "test" / "queries_candidates.jsonl"
OUTPUT_DIR = ROOT / "data" / "benchmark_governed_v0_1" / "test"
HUMAN_TEMPLATE = OUTPUT_DIR / "human_written_queries_template.tsv"
DOMAIN_REPORT = OUTPUT_DIR / "domain_balance_report.json"

# Targets from FREEZE_CRITERIA
TARGET_QUERIES_PER_DOMAIN = 50
MIN_HUMAN_RATIO = 0.4

def main():
    # Load candidate queries
    queries_by_domain = Counter()
    queries_by_source = Counter()
    all_queries = []
    
    with CANDIDATE_QUERIES.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                domain = q.get("domain", "unknown")
                source = q.get("source", "unknown")
                queries_by_domain[domain] += 1
                queries_by_source[source] += 1
                all_queries.append(q)
    
    print("=" * 60)
    print("CURRENT QUERY DISTRIBUTION")
    print("=" * 60)
    print(f"\nBy Domain:")
    for domain, count in sorted(queries_by_domain.items()):
        status = "OK" if count >= TARGET_QUERIES_PER_DOMAIN else f"NEED {TARGET_QUERIES_PER_DOMAIN - count} MORE"
        print(f"  {domain:<15} {count:>4} / {TARGET_QUERIES_PER_DOMAIN}  {status}")
    
    print(f"\nBy Source:")
    for source, count in sorted(queries_by_source.items()):
        pct = count / len(all_queries) * 100
        print(f"  {source:<25} {count:>4} ({pct:.1f}%)")
    
    human_count = queries_by_source.get("human_written", 0)
    total = len(all_queries)
    human_ratio = human_count / total if total else 0
    print(f"\nHuman-written: {human_count}/{total} = {human_ratio:.1%} (target: {MIN_HUMAN_RATIO:.0%})")
    if human_ratio < MIN_HUMAN_RATIO:
        needed = int(MIN_HUMAN_RATIO * total) - human_count
        print(f"  -> NEED {needed} MORE human-written queries")
    
    # Create human-written template
    print(f"\n[*] Creating human-written template at {HUMAN_TEMPLATE}")
    
    columns = [
        "query_id",
        "query_text",
        "domain",
        "query_type",
        "target_page_ids",
        "hardness_level",
        "source",
        "created_by",
        "created_at",
        "notes",
    ]
    
    # Suggest queries for domains that need more
    needed_by_domain = {}
    for domain, count in queries_by_domain.items():
        if count < TARGET_QUERIES_PER_DOMAIN:
            needed_by_domain[domain] = TARGET_QUERIES_PER_DOMAIN - count
    
    with HUMAN_TEMPLATE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        
        # Example rows for each domain
        examples = {
            "legal": [
                {"query_text": "Mức phạt vi phạm an toàn lao động theo Luật 2019 là bao nhiêu?", "query_type": "legal_clause", "hardness": "medium"},
                {"query_text": "Quy trình giải quyết khiếu nại vi phạm pháp luật lao động", "query_type": "fact_lookup", "hardness": "easy"},
                {"query_text": "Điều khoản về thời gian làm việc thêm giờ mới nhất", "query_type": "legal_clause", "hardness": "medium"},
            ],
            "financial": [
                {"query_text": "Tổng thu ngân sách nhà nước năm 2023 đạt bao nhiêu tỷ đồng?", "query_type": "numeric_table", "hardness": "easy"},
                {"query_text": "So sánh tăng trưởng GDP quý 1 và quý 2 năm 2024", "query_type": "multi_cell_comparison", "hardness": "hard"},
                {"query_text": "Cân đối kế hoạch đầu tư công năm 2023 bộ phận trung ương", "query_type": "numeric_table", "hardness": "medium"},
            ],
            "education": [
                {"query_text": "Định nghĩa thuật toán Dijkstra và độ phức tạp thời gian", "query_type": "fact_lookup", "hardness": "easy"},
                {"query_text": "Khác biệt giữa stack và queue trong cấu trúc dữ liệu", "query_type": "paraphrase_or_abbreviation", "hardness": "medium"},
                {"query_text": "Mô hình OSI 7 tầng và chức năng từng tầng", "query_type": "fact_lookup", "hardness": "medium"},
            ],
            "healthcare": [
                {"query_text": "Phác đồ điều trị tiểu đường typ 2 theo hướng dẫn Bộ Y tế 2023", "query_type": "fact_lookup", "hardness": "medium"},
                {"query_text": "Triệu chứng và chẩn đoán suy tim mạn tính", "query_type": "fact_lookup", "hardness": "easy"},
                {"query_text": "Liều lượng paracetamol cho trẻ em theo cân nặng", "query_type": "numeric_table", "hardness": "easy"},
            ],
        }
        
        query_counter = 0
        for domain, needed in needed_by_domain.items():
            domain_examples = examples.get(domain, [])
            for i in range(needed):
                if i < len(domain_examples):
                    ex = domain_examples[i]
                else:
                    ex = {"query_text": f"[TODO: Write {domain} query #{i+1}]", "query_type": "fact_lookup", "hardness": "medium"}
                
                query_counter += 1
                qid = f"q_human_{domain}_{query_counter:03d}"
                writer.writerow({
                    "query_id": qid,
                    "query_text": ex["query_text"],
                    "domain": domain,
                    "query_type": ex["query_type"],
                    "target_page_ids": "",
                    "hardness_level": ex["hardness"],
                    "source": "human_written",
                    "created_by": "",
                    "created_at": "",
                    "notes": "Target page IDs to be filled after pooling",
                })
    
    print(f"[+] Template created with {query_counter} suggested queries")
    
    # Save domain balance report
    report = {
        "current_distribution": {
            "by_domain": dict(queries_by_domain),
            "by_source": dict(queries_by_source),
            "total": total,
            "human_ratio": round(human_ratio, 4),
        },
        "targets": {
            "min_queries_per_domain": TARGET_QUERIES_PER_DOMAIN,
            "min_human_ratio": MIN_HUMAN_RATIO,
        },
        "gaps": {
            "queries_per_domain": {d: max(0, TARGET_QUERIES_PER_DOMAIN - c) for d, c in queries_by_domain.items()},
            "human_written_needed": max(0, int(MIN_HUMAN_RATIO * total) - human_count),
        },
        "template_created": str(HUMAN_TEMPLATE),
        "example_queries_provided": len(needed_by_domain) > 0,
    }
    
    DOMAIN_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Domain balance report saved to {DOMAIN_REPORT}")

if __name__ == "__main__":
    main()