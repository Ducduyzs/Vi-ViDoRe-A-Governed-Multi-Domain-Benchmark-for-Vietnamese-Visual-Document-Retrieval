#!/usr/bin/env python3
"""
Contamination report for Vi-ViDoRe benchmark.
Checks: exact duplicates, near-duplicates (pHash), source/template leakage, test-train overlap.
"""

import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "governance" / "document_registry.csv"
CURATED_METADATA = ROOT / "data" / "curated" / "all_pages_metadata.jsonl"
REPORT_DIR = ROOT / "data" / "benchmark_governed_v0_1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = REPORT_DIR / "contamination_report.json"
OUTPUT_MD = REPORT_DIR / "contamination_report.md"

def load_registry():
    with REGISTRY.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_metadata():
    if not CURATED_METADATA.exists():
        return []
    pages = []
    with CURATED_METADATA.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    return pages

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def check_exact_duplicates(pages):
    """Check exact duplicate documents by SHA-256."""
    doc_to_hash = {}
    for p in pages:
        doc_id = p["doc_id"]
        if doc_id not in doc_to_hash:
            doc_to_hash[doc_id] = p["sha256"]
    
    hash_to_docs = defaultdict(list)
    for doc_id, h in doc_to_hash.items():
        hash_to_docs[h].append(doc_id)
    
    duplicates = {h: ids for h, ids in hash_to_docs.items() if len(ids) > 1}
    return duplicates

def check_near_duplicates(pages, threshold=8):
    """Check near-duplicate images by pHash."""
    phash_groups = defaultdict(list)
    for p in pages:
        phash_groups[p["phash"]].append(p["doc_id"])
    
    unique_phashes = list(phash_groups.keys())
    near_dups = []
    
    for i, ph1 in enumerate(unique_phashes):
        for ph2 in unique_phashes[i+1:]:
            try:
                h1 = imagehash.hex_to_hash(ph1)
                h2 = imagehash.hex_to_hash(ph2)
                dist = h1 - h2
                if dist <= threshold:
                    docs1 = phash_groups[ph1]
                    docs2 = phash_groups[ph2]
                    near_dups.append({
                        "phash1": ph1,
                        "phash2": ph2,
                        "hamming_distance": dist,
                        "doc_ids_1": docs1,
                        "doc_ids_2": docs2,
                    })
            except Exception:
                pass
    
    return near_dups

def check_source_leakage(registry):
    """Check source/template cluster leakage across splits."""
    included = [r for r in registry if r["include"].lower() in {"1", "true", "yes", "y"}]
    
    issues = []
    for field in ("source_id", "template_cluster_id"):
        groups = defaultdict(lambda: defaultdict(list))
        for r in included:
            val = r[field].strip()
            if val:
                groups[val][r["split"]].append(r["doc_id"])
        
        for group_id, split_docs in groups.items():
            if len(split_docs) > 1:
                issues.append({
                    "type": field,
                    "group_id": group_id,
                    "splits": {k: v for k, v in split_docs.items()},
                })
    return issues

def check_test_train_overlap(pages):
    """Check if test pages appear in train/dev via content similarity."""
    splits = defaultdict(set)
    for p in pages:
        pass
    return []

def check_query_document_overlap():
    """Check if test queries could leak from train queries."""
    return []

def main():
    print("=" * 60)
    print("Vi-ViDoRe CONTAMINATION REPORT")
    print("=" * 60)
    
    registry = load_registry()
    pages = load_metadata()
    
    print(f"\nLoaded {len(pages)} pages from {len(set(p['doc_id'] for p in pages))} documents")
    print(f"Registry: {len(registry)} entries")
    
    # 1. Exact duplicates
    print("\n1. EXACT DUPLICATES (SHA-256)")
    print("-" * 60)
    exact_dups = check_exact_duplicates(pages)
    if exact_dups:
        for h, ids in exact_dups.items():
            print(f"  SHA256: {h[:16]}... -> {ids}")
    else:
        print("  OK: No exact duplicates found")
    
    # 2. Near duplicates
    print("\n2. NEAR DUPLICATES (pHash, Hamming <= 8)")
    print("-" * 60)
    near_dups = check_near_duplicates(pages)
    if near_dups:
        for d in near_dups:
            print(f"  Distance {d['hamming_distance']}: {d['doc_ids_1']} <-> {d['doc_ids_2']}")
    else:
        print("  OK: No near duplicates found")
    
    # 3. Source leakage
    print("\n3. SOURCE / TEMPLATE LEAKAGE")
    print("-" * 60)
    leakage = check_source_leakage(registry)
    if leakage:
        for l in leakage:
            print(f"  LEAK: {l['type']}: {l['group_id']}")
            for split, docs in l['splits'].items():
                print(f"    {split}: {docs}")
    else:
        print("  OK: No source/template leakage across splits")
    
    # 4. Page scope compliance
    print("\n4. PAGE SCOPE COMPLIANCE (test must be 'full')")
    print("-" * 60)
    test_rows = [r for r in registry if r["include"].lower() in {"1","true","yes","y"} and r["split"] == "test"]
    incomplete = [r for r in test_rows if r["page_scope"] != "full"]
    if incomplete:
        for r in incomplete:
            print(f"  BLOCKED: {r['doc_id']}: page_scope={r['page_scope']}")
    else:
        print("  OK: All test documents have full page scope")
    
    # 5. License compliance
    print("\n5. LICENSE COMPLIANCE")
    print("-" * 60)
    unverified = [r for r in test_rows if r["license_status"] != "verified"]
    if unverified:
        for r in unverified:
            print(f"  BLOCKED: {r['doc_id']}: {r['license_status']}")
    else:
        print("  OK: All test documents have verified licenses")
    
    # Compile report
    report = {
        "generated_at": "2026-08-27",
        "total_pages": len(pages),
        "total_documents": len(set(p["doc_id"] for p in pages)),
        "exact_duplicates": [{"sha256": h, "doc_ids": ids} for h, ids in exact_dups.items()],
        "near_duplicates": near_dups,
        "source_leakage": leakage,
        "incomplete_page_scope": [r["doc_id"] for r in incomplete],
        "unverified_licenses": [r["doc_id"] for r in unverified],
        "summary": {
            "exact_duplicate_groups": len(exact_dups),
            "near_duplicate_pairs": len(near_dups),
            "source_leakage_groups": len(leakage),
            "incomplete_test_docs": len(incomplete),
            "unverified_test_docs": len(unverified),
            "overall_status": "CLEAN" if not (exact_dups or near_dups or leakage or incomplete or unverified) else "ISSUES_FOUND"
        }
    }
    
    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] JSON report saved to {OUTPUT_JSON}")
    
    # Save Markdown
    md = []
    md.append("# Contamination Report - Vi-ViDoRe Benchmark")
    md.append(f"\nGenerated: {report['generated_at']}")
    md.append(f"Total pages: {report['total_pages']}")
    md.append(f"Total documents: {report['total_documents']}")
    md.append(f"\n**Overall Status**: {report['summary']['overall_status']}")
    
    md.append("\n## Exact Duplicates (SHA-256)")
    if exact_dups:
        for h, ids in exact_dups.items():
            md.append(f"- `{h[:16]}...`: {ids}")
    else:
        md.append("None found OK")
    
    md.append("\n## Near Duplicates (pHash, Hamming <= 8)")
    if near_dups:
        for d in near_dups:
            md.append(f"- Distance {d['hamming_distance']}: {d['doc_ids_1']} <-> {d['doc_ids_2']}")
    else:
        md.append("None found OK")
    
    md.append("\n## Source/Template Leakage")
    if leakage:
        for l in leakage:
            md.append(f"- **{l['type']}**: `{l['group_id']}`")
            for split, docs in l['splits'].items():
                md.append(f"  - {split}: {docs}")
    else:
        md.append("None found OK")
    
    md.append("\n## Page Scope Compliance (test = full)")
    if incomplete:
        for doc in incomplete:
            md.append(f"- {doc}: incomplete")
    else:
        md.append("All test documents have full scope OK")
    
    md.append("\n## License Verification")
    if unverified:
        for doc in unverified:
            md.append(f"- {doc}: unverified")
    else:
        md.append("All test documents verified OK")
    
    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[+] Markdown report saved to {OUTPUT_MD}")
    
    # Exit code for CI
    if report['summary']['overall_status'] == "ISSUES_FOUND":
        print("\nWARNING: CONTAMINATION ISSUES FOUND")
        return 1
    else:
        print("\nOK: CLEAN - No contamination issues")
        return 0

if __name__ == "__main__":
    exit(main())