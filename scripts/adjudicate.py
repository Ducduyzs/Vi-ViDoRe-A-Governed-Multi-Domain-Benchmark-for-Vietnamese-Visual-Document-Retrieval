#!/usr/bin/env python3
"""
Adjudication script for double annotation.
Computes Cohen's kappa, performs majority vote, outputs final qrels.
"""

import csv
import json
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parent.parent

def cohens_kappa(annotations_a, annotations_b, labels=(0, 1, 2)):
    """
    Compute Cohen's kappa for two annotators.
    annotations_a/b: dict[(query_id, page_id)] -> relevance (0/1/2)
    """
    # Get common pairs
    common_keys = set(annotations_a.keys()) & set(annotations_b.keys())
    if not common_keys:
        return 0.0, 0, {}
    
    n = len(common_keys)
    agree = sum(1 for k in common_keys if annotations_a[k] == annotations_b[k])
    p_o = agree / n
    
    # Expected agreement
    counts_a = Counter(annotations_a[k] for k in common_keys)
    counts_b = Counter(annotations_b[k] for k in common_keys)
    
    p_e = sum((counts_a.get(l, 0) / n) * (counts_b.get(l, 0) / n) for l in labels)
    
    if p_e == 1.0:
        return 1.0, n, {"agreement": p_o, "expected": p_e}
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, n, {"agreement": p_o, "expected": p_e, "per_label": dict(counts_a)}

def load_annotations(tsv_path, annotator_id=None):
    """Load annotations from TSV file.
    If annotator_id is provided, only load annotations for that annotator.
    Otherwise, load all annotations and group by annotator.
    """
    annotations = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["relevance"] in {"0", "1", "2"}:
                if annotator_id is None or row["annotator_id"] == annotator_id:
                    key = (row["query_id"], row["page_id"])
                    annotations[key] = int(row["relevance"])
    return annotations

def adjudicate(tsv_path_a, tsv_path_b, output_tsv, output_report):
    """Main adjudication function."""
    print(f"Loading annotations from {tsv_path_a} and {tsv_path_b}")
    
    # Extract annotator IDs from filenames or use defaults
    ann_a = Path(tsv_path_a).stem.split("_")[-1] if "_" in Path(tsv_path_a).stem else "annotator_A"
    ann_b = Path(tsv_path_b).stem.split("_")[-1] if "_" in Path(tsv_path_b).stem else "annotator_B"
    
    # Load annotations
    annotations_a = load_annotations(Path(tsv_path_a), ann_a)
    annotations_b = load_annotations(Path(tsv_path_b), ann_b)
    
    print(f"  {ann_a}: {len(annotations_a)} judgments")
    print(f"  {ann_b}: {len(annotations_b)} judgments")
    
    # Compute Cohen's kappa
    kappa, n_common, stats = cohens_kappa(annotations_a, annotations_b)
    print(f"\nCohen's kappa: {kappa:.4f} (n={n_common} common pairs)")
    print(f"  Observed agreement: {stats['agreement']:.4f}")
    print(f"  Expected agreement: {stats['expected']:.4f}")
    
    if kappa < 0.67:
        print(f"\n⚠️  WARNING: Kappa {kappa:.4f} < 0.67 threshold!")
        print("   Need adjudicator review. Run with --adjudicator flag or manual review.")
    
    # Merge: for each pair, take majority vote (or max for safety)
    all_keys = set(annotations_a.keys()) | set(annotations_b.keys())
    final_annotations = {}
    disagreements = []
    
    for key in all_keys:
        val_a = annotations_a.get(key)
        val_b = annotations_b.get(key)
        
        if val_a is not None and val_b is not None:
            if val_a == val_b:
                final_annotations[key] = val_a
            else:
                # Disagreement - take max (conservative for recall)
                final_annotations[key] = max(val_a, val_b)
                disagreements.append({
                    "query_id": key[0],
                    "page_id": key[1],
                    f"{ann_a}": val_a,
                    f"{ann_b}": val_b,
                    "resolved": max(val_a, val_b),
                })
        elif val_a is not None:
            final_annotations[key] = val_a
        elif val_b is not None:
            final_annotations[key] = val_b
    
    print(f"\nTotal unique pairs: {len(all_keys)}")
    print(f"Agreements: {len(all_keys) - len(disagreements)}")
    print(f"Disagreements resolved (max): {len(disagreements)}")
    
    # Write final qrels TSV
    with output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "page_id", "relevance"], delimiter="\t")
        writer.writeheader()
        for (qid, pid), rel in sorted(final_annotations.items()):
            writer.writerow({"query_id": qid, "page_id": pid, "relevance": rel})
    
    print(f"\n[+] Final qrels written to {output_tsv}")
    
    # Write adjudication report
    report = {
        "kappa": round(kappa, 4),
        "n_common_pairs": n_common,
        "observed_agreement": round(stats["agreement"], 4),
        "expected_agreement": round(stats["expected"], 4),
        "total_pairs": len(all_keys),
        "agreements": len(all_keys) - len(disagreements),
        "disagreements": len(disagreements),
        "resolution_method": "max (conservative for recall)",
        "disagreement_details": disagreements[:50],  # First 50
        "annotator_a": ann_a,
        "annotator_b": ann_b,
        "threshold_passed": kappa >= 0.67,
    }
    
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Adjudication report written to {output_report}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python adjudicate.py <annotator_A.tsv> <annotator_B.tsv> <output_qrels.tsv> [output_report.json]")
        sys.exit(1)
    
    tsv_a = Path(sys.argv[1])
    tsv_b = Path(sys.argv[2])
    out_tsv = Path(sys.argv[3])
    out_report = Path(sys.argv[4]) if len(sys.argv) > 4 else out_tsv.with_suffix(".adjudication_report.json")
    
    adjudicate(tsv_a, tsv_b, out_tsv, out_report)