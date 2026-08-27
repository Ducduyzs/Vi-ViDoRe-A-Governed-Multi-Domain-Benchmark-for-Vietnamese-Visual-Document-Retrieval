#!/usr/bin/env python3
"""
Pilot 100 workflow tracker.
Tracks annotation progress, computes kappa, manages assignments.
"""

import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = ROOT / "data" / "benchmark_governed_v0_1" / "test"
MANIFEST = ANNOTATION_DIR / "pilot100_manifest.json"

def load_manifest():
    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_annotations(tsv_path):
    """Load annotations from TSV."""
    annotations = {}
    if not tsv_path.exists():
        return annotations
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["annotator_id"] and row["relevance"] in {"0", "1", "2"}:
                key = (row["query_id"], row["page_id"])
                annotations[key] = {
                    "annotator": row["annotator_id"],
                    "relevance": int(row["relevance"]),
                    "status": row["query_status"],
                    "note": row["evidence_note"],
                    "judged_at": row["judged_at"],
                }
    return annotations

def track_progress():
    manifest = load_manifest()
    total_pairs = manifest.get("total_pairs", 0)
    query_ids = manifest.get("query_ids", [])
    
    # Find all annotation files
    annotation_files = list(ANNOTATION_DIR.glob("annotations_*.tsv"))
    
    print("=" * 60)
    print("PILOT 100 PROGRESS TRACKER")
    print("=" * 60)
    print(f"Target: {len(query_ids)} queries, ~{total_pairs} pairs")
    print(f"Guideline version: {manifest.get('guideline_version', 'unknown')}")
    print(f"Created: {manifest.get('created_at', 'unknown')}")
    print()
    
    all_annotations = {}
    annotator_stats = Counter()
    
    for ann_file in annotation_files:
        annotations = load_annotations(ann_file)
        for key, ann in annotations.items():
            if key not in all_annotations:
                all_annotations[key] = []
            all_annotations[key].append(ann)
            annotator_stats[ann["annotator"]] += 1
    
    # Overall stats
    total_judged = sum(len(v) for v in all_annotations.values())
    unique_pairs = len(all_annotations)
    
    # Per annotator
    print("ANNOTATOR PROGRESS:")
    for ann, count in annotator_stats.most_common():
        print(f"  {ann}: {count} judgments")
    
    print(f"\nTOTAL: {total_judged} judgments on {unique_pairs} unique pairs")
    print(f"COVERAGE: {unique_pairs}/{total_pairs} = {unique_pairs/total_pairs*100:.1f}%" if total_pairs else "COVERAGE: N/A")
    
    # Agreement analysis
    pairs_with_multiple = {k: v for k, v in all_annotations.items() if len(v) >= 2}
    if pairs_with_multiple:
        agreements = sum(1 for v in pairs_with_multiple.values() if len(set(a["relevance"] for a in v)) == 1)
        print(f"\nAGREEMENT on {len(pairs_with_multiple)} double-annotated pairs:")
        print(f"  Full agreement: {agreements} ({agreements/len(pairs_with_multiple)*100:.1f}%)")
        
        # Per-annotator pair agreement
        annotators = list(annotator_stats.keys())
        if len(annotators) >= 2:
            from scripts.adjudicate import cohens_kappa
            ann_a = {k: v[0]["relevance"] for k, v in pairs_with_multiple.items()}
            ann_b = {k: v[1]["relevance"] for k, v in pairs_with_multiple.items()}
            kappa, n, stats = cohens_kappa(ann_a, ann_b)
            print(f"  Cohen's kappa: {kappa:.4f} (n={n})")
            print(f"  Threshold (0.67): {'PASS' if kappa >= 0.67 else 'FAIL'}")
    
    # Per-query progress
    query_progress = Counter()
    for (qid, _), anns in all_annotations.items():
        query_progress[qid] += 1
    
    fully_annotated = sum(1 for qid, count in query_progress.items() if count >= 2)  # At least 2 judgments per query
    print(f"\nPER-QUERY: {fully_annotated}/{len(query_ids)} queries have >=2 judgments")
    
    # Save progress report
    report = {
        "timestamp": datetime.now().isoformat(),
        "manifest": manifest,
        "annotator_stats": dict(annotator_stats),
        "total_judgments": total_judged,
        "unique_pairs_annotated": unique_pairs,
        "coverage_pct": round(unique_pairs/total_pairs*100, 1) if total_pairs else 0,
        "queries_fully_annotated": fully_annotated,
        "total_queries": len(query_ids),
    }
    
    if pairs_with_multiple:
        from scripts.adjudicate import cohens_kappa
        ann_a = {k: v[0]["relevance"] for k, v in pairs_with_multiple.items()}
        ann_b = {k: v[1]["relevance"] for k, v in pairs_with_multiple.items()}
        kappa, n, stats = cohens_kappa(ann_a, ann_b)
        report["kappa"] = round(kappa, 4)
        report["kappa_threshold_passed"] = kappa >= 0.67
    
    report_path = ANNOTATION_DIR / f"pilot100_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Progress report saved to {report_path}")

def assign_queries(annotator_id, num_queries=20):
    """Assign queries to an annotator."""
    manifest = load_manifest()
    query_ids = manifest.get("query_ids", [])
    
    # Load existing assignments
    assigned = set()
    for ann_file in ANNOTATION_DIR.glob("annotations_*.tsv"):
        with ann_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["annotator_id"]:
                    assigned.add(row["query_id"])
    
    available = [q for q in query_ids if q not in assigned]
    selected = available[:num_queries]
    
    print(f"Assigned {len(selected)} queries to {annotator_id}")
    for q in selected:
        print(f"  {q}")
    
    return selected

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "assign":
        annotator = sys.argv[2] if len(sys.argv) > 2 else "annotator_X"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        assign_queries(annotator, n)
    else:
        track_progress()