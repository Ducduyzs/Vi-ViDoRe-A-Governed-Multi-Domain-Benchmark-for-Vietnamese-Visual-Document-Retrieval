"""Rank scan/table/chart/form pages for human review.

Heuristic suggestions are never authoritative benchmark labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/benchmark_governed_v0_1/test/pages_metadata.jsonl"
DEFAULT_OUTPUT = ROOT / "data/benchmark_governed_v0_1/review/page_type_candidates.csv"
LABELS = ("scanned", "table_heavy", "chart_heavy", "form_or_template")


def repair_mojibake(text: str) -> str:
    """Best-effort repair for UTF-8 text decoded as latin-1."""
    if not text:
        return ""
    if any(marker in text for marker in ("Ã", "Ä", "Æ", "á»", "â€")):
        try:
            repaired = text.encode(bytes((99, 112, 49, 50, 53, 50)).decode()).decode()
            if repaired.count(chr(65533)) <= text.count(chr(65533)):
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", repair_mojibake(text).lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def keyword_hits(text: str, phrases: Iterable[str]) -> List[str]:
    normalized = fold(text)
    return sorted({phrase for phrase in phrases if phrase in normalized})


def image_layout_features(image_path: str) -> Dict[str, float]:
    """Compute cheap layout signals without OCR or model dependencies."""
    path = Path(image_path)
    empty = {
        "image_available": 0.0,
        "ink_ratio": 0.0,
        "horizontal_line_ratio": 0.0,
        "vertical_line_ratio": 0.0,
        "large_dark_block_ratio": 0.0,
    }
    if not path.exists():
        return empty
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((700, 700), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.uint8)

    dark = arr < 185
    height, width = dark.shape
    horizontal = float((dark.mean(axis=1) >= 0.42).sum() / max(1, height))
    vertical = float((dark.mean(axis=0) >= 0.42).sum() / max(1, width))
    block_h, block_w = max(1, height // 12), max(1, width // 12)
    dense_blocks = total_blocks = 0
    for y in range(0, height, block_h):
        for x in range(0, width, block_w):
            block = dark[y : y + block_h, x : x + block_w]
            if block.size:
                total_blocks += 1
                dense_blocks += int(float(block.mean()) >= 0.28)
    return {
        "image_available": 1.0,
        "ink_ratio": round(float(dark.mean()), 6),
        "horizontal_line_ratio": round(horizontal, 6),
        "vertical_line_ratio": round(vertical, 6),
        "large_dark_block_ratio": round(dense_blocks / max(1, total_blocks), 6),
    }


def classify_page(page: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, float]]:
    text = str(page.get("native_text", ""))
    normalized = fold(text)
    chars = int(page.get("char_count", len(text.strip())))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    numeric_lines = sum(bool(re.search(r"(?:\d[\s.,:%-]*){3,}", line)) for line in lines)
    aligned_lines = sum(bool(re.search(r"\S\s{3,}\S", line)) for line in lines)
    features = image_layout_features(str(page.get("image_path", "")))

    table_hits = keyword_hits(text, ("bang ", "bang:", "bieu bang", "don vi tinh", "tong cong"))
    chart_hits = keyword_hits(
        text, ("bieu do", "do thi", "hinh ", "figure", "chart", "truc tung", "truc hoanh", "chu giai")
    )
    form_hits = keyword_hits(
        text,
        (
            "bieu mau", "mau so", "don de nghi", "don dang ky", "to khai",
            "phieu ", "ho va ten", "ngay sinh", "dia chi", "ky ten", "nguoi khai",
            "danh dau", "xac nhan", "cong hoa xa hoi chu nghia viet nam",
        ),
    )
    checkbox_hits = len(re.findall(r"(?:\[\s?\]|□|☐|☑|[_\.]{5,})", text))

    scan_score = 0.75 if str(page.get("source_type", "")) == "scanned" else 0.0
    scan_score += 0.20 if chars < 50 else 0.0
    scan_score += 0.05 if chars < 15 and features["ink_ratio"] > 0.015 else 0.0
    table_score = min(
        1.0,
        0.12 * len(table_hits) + min(0.35, numeric_lines * 0.035)
        + min(0.18, aligned_lines * 0.03) + min(0.15, digit_ratio * 1.5)
        + min(0.20, (features["horizontal_line_ratio"] + features["vertical_line_ratio"]) * 6.0),
    )
    chart_score = min(
        1.0,
        0.13 * len(chart_hits) + (0.18 if "%" in normalized else 0.0)
        + (0.18 if 20 <= chars <= 700 else 0.0)
        + min(0.28, features["large_dark_block_ratio"] * 1.4)
        + (0.12 if digit_ratio >= 0.08 else 0.0),
    )
    form_score = min(
        1.0,
        0.13 * len(form_hits) + min(0.28, checkbox_hits * 0.07)
        + min(0.20, features["horizontal_line_ratio"] * 7.0)
        + (0.12 if 40 <= chars <= 1500 else 0.0),
    )
    scores = {
        "scanned": round(min(1.0, scan_score), 4),
        "table_heavy": round(table_score, 4),
        "chart_heavy": round(chart_score, 4),
        "form_or_template": round(form_score, 4),
    }
    evidence = {
        "table_keywords": table_hits,
        "chart_keywords": chart_hits,
        "form_keywords": form_hits,
        "checkbox_or_blank_count": [str(checkbox_hits)] if checkbox_hits else [],
    }
    return scores, evidence, features


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(pages: Iterable[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in pages:
        scores, evidence, features = classify_page(page)
        ranked = sorted(LABELS, key=lambda label: (-scores[label], label))
        candidates = [label for label in ranked if scores[label] >= threshold]
        existing = str(page.get("page_type", ""))
        if existing in LABELS and existing not in candidates:
            candidates.append(existing)
        if str(page.get("source_type", "")) == "scanned" and "scanned" not in candidates:
            candidates.append("scanned")
        if not candidates:
            continue
        evidence_parts = [
            f"{key}={','.join(values)}" for key, values in evidence.items() if values
        ]
        evidence_parts += [
            f"chars={page.get('char_count', 0)}", f"ink={features['ink_ratio']}",
            f"hline={features['horizontal_line_ratio']}", f"vline={features['vertical_line_ratio']}",
        ]
        rows.append({
            "page_id": page["page_id"], "doc_id": page["doc_id"],
            "page_num": page["page_num"], "domain": page.get("domain", ""),
            "image_path": page.get("image_path", ""), "existing_page_type": existing,
            "existing_source_type": page.get("source_type", ""),
            "suggested_labels": "|".join(candidates), "top_suggestion": ranked[0],
            "top_score": scores[ranked[0]],
            **{f"score_{label}": scores[label] for label in LABELS},
            "evidence": "; ".join(evidence_parts), "human_label": "",
            "reviewer_id": "", "review_status": "PENDING", "review_note": "", "reviewed_at": "",
        })
    return sorted(rows, key=lambda row: (-float(row["top_score"]), row["page_id"]))


def write_outputs(rows: List[Dict[str, Any]], output: Path, input_path: Path, threshold: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else [
        "page_id", "doc_id", "page_num", "domain", "image_path", "existing_page_type",
        "existing_source_type", "suggested_labels", "top_suggestion", "top_score",
        *(f"score_{label}" for label in LABELS), "evidence", "human_label",
        "reviewer_id", "review_status", "review_note", "reviewed_at",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(filter(None, str(row["suggested_labels"]).split("|")))
    summary = {
        "status": "REVIEW_REQUIRED",
        "warning": "Heuristic suggestions are not benchmark labels until human review.",
        "input": str(input_path.resolve()), "output": str(output.resolve()),
        "threshold": threshold, "candidate_pages": len(rows),
        "suggested_label_counts": dict(sorted(counts.items())),
        "review_status_counts": {"PENDING": len(rows)},
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.with_suffix(".README.md").write_text(
        "# Page-type review queue\n\n"
        "Heuristic candidates only; this script does not modify metadata. Review each image and set "
        "`human_label` to `scanned`, `table_heavy`, `chart_heavy`, `form_or_template`, "
        "`text_heavy`, `mixed`, or `exclude`. Set `review_status=APPROVED` and complete "
        "reviewer/time fields. A second reviewer must verify pages used by test queries.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    rows = build_rows(load_jsonl(args.input), args.threshold)
    write_outputs(rows, args.output, args.input, args.threshold)
    print(f"[REVIEW_REQUIRED] Wrote {len(rows)} candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
