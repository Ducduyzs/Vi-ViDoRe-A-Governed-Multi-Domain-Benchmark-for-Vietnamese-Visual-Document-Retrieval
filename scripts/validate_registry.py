#!/usr/bin/env python3
"""
Registry validation and helper to find new source documents.
Checks: source diversity, template leakage, domain balance.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "governance" / "document_registry.csv"
CRITERIA = ROOT / "data" / "governance" / "FREEZE_CRITERIA.json"

def load_registry():
    with REGISTRY.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_criteria():
    import json
    with CRITERIA.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    rows = load_registry()
    criteria = load_criteria()
    
    included = [r for r in rows if r["include"].lower() in {"1", "true", "yes", "y"}]
    test_rows = [r for r in included if r["split"] == "test"]
    
    print("=" * 80)
    print("REGISTRY VALIDATION - TEST SPLIT")
    print("=" * 80)
    
    # 1. Source diversity per domain
    print("\n1. SOURCE DIVERSITY (min 2 per domain)")
    print("-" * 80)
    source_by_domain = defaultdict(set)
    for r in test_rows:
        source_by_domain[r["domain"]].add(r["source_id"])
    
    min_sources = criteria["min_test_source_groups_per_domain"]
    for domain in criteria["required_test_domains"]:
        sources = source_by_domain.get(domain, set())
        status = "OK" if len(sources) >= min_sources else "BLOCKED"
        print(f"  {domain:<15} {len(sources)}/{min_sources} sources {status}")
        for s in sorted(sources):
            print(f"    - {s}")
    
    # 2. Template cluster leakage
    print("\n2. TEMPLATE CLUSTER LEAKAGE CHECK")
    print("-" * 80)
    clusters = defaultdict(lambda: defaultdict(list))
    for r in included:
        cid = r["template_cluster_id"]
        if cid:
            clusters[cid][r["split"]].append(r["doc_id"])
    
    leaked = False
    for cid, splits in clusters.items():
        if len(splits) > 1:
            leaked = True
            print(f"  LEAK: cluster '{cid}' spans splits:")
            for split, docs in splits.items():
                print(f"    {split}: {docs}")
    if not leaked:
        print("  OK: No template cluster leakage across splits")
    
    # 3. Page scope
    print("\n3. PAGE SCOPE (test must be 'full')")
    print("-" * 80)
    incomplete = [r for r in test_rows if r["page_scope"] != "full"]
    if incomplete:
        for r in incomplete:
            print(f"  BLOCKED: {r['doc_id']}: page_scope={r['page_scope']}")
    else:
        print("  OK: All test documents have full page scope")
    
    # 4. License status
    print("\n4. LICENSE STATUS (test must be verified)")
    print("-" * 80)
    unverified = [r for r in test_rows if r["license_status"] != "verified"]
    if unverified:
        for r in unverified:
            print(f"  BLOCKED: {r['doc_id']}: license_status={r['license_status']}, redistribution={r['redistribution_allowed']}")
    else:
        print("  OK: All test documents verified")
    
    # 5. Domain page coverage
    print("\n5. DOMAIN PAGE COVERAGE (min 10 pages per domain)")
    print("-" * 80)
    # Need to load curated metadata for page counts
    curated_path = ROOT / "data" / "curated" / "all_pages_metadata.jsonl"
    if curated_path.exists():
        import json
        pages = [json.loads(line) for line in curated_path.open(encoding="utf-8") if line.strip()]
        doc_to_pages = defaultdict(list)
        for p in pages:
            doc_to_pages[p["doc_id"]].append(p)
        
        test_doc_ids = {r["doc_id"] for r in test_rows}
        pages_by_domain = Counter()
        for doc_id in test_doc_ids:
            for p in doc_to_pages.get(doc_id, []):
                pages_by_domain[p["domain"]] += 1
        
        min_pages = criteria["min_test_pages_per_domain"]
        for domain in criteria["required_test_domains"]:
            count = pages_by_domain.get(domain, 0)
            status = "OK" if count >= min_pages else "BLOCKED"
            print(f"  {domain:<15} {count}/{min_pages} pages {status}")
    
    # 6. Missing fields report
    print("\n6. MISSING CRITICAL FIELDS")
    print("-" * 80)
    critical_fields = ["source_url", "license_name", "license_url", "publisher"]
    for r in test_rows:
        missing = [f for f in critical_fields if not r[f].strip()]
        if missing:
            print(f"  {r['doc_id']}: missing {missing}")
    
    # 7. Suggestions for new sources
    print("\n7. SUGGESTED ACTIONS FOR SOURCE DIVERSITY")
    print("-" * 80)
    for domain in criteria["required_test_domains"]:
        sources = source_by_domain.get(domain, set())
        needed = max(0, min_sources - len(sources))
        if needed > 0:
            print(f"  {domain}: Need {needed} MORE independent source(s)")
            if domain == "legal":
                print(f"    -> Find legal docs from: Bo Tu phap, Luat Viet Nam, Nghi dinh/Thong tu khac")
                print(f"    -> Or: University law textbooks from different publishers")
            elif domain == "financial":
                print(f"    -> Find financial docs from: World Bank (other reports), IMF, ADB, VN Ministry of Finance")
                print(f"    -> Or: Annual reports from different VN banks/companies (with permission)")
            elif domain == "education":
                print(f"    -> Already have 3 sources (OK)")
            elif domain == "healthcare":
                print(f"    -> Already have 2 sources (OK)")

if __name__ == "__main__":
    main()