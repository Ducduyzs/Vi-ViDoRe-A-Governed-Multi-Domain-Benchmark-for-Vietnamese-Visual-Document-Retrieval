"""Build a governed Vietnamese benchmark candidate and enforce freeze gates.

This script never mutates the legacy benchmark. It combines reviewed legacy
metadata with newly cleared PDFs, applies the authoritative document registry,
writes candidate splits for annotation, and refuses an official freeze until
all scientific and legal gates pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.data.pdf_processor import PDFProcessor
from src.data.schema import DomainType, PageMetadata
from src.data.query_generator import QueryGenerator
from src.data.query_sanitizer import QuerySanitizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "data" / "governance" / "document_registry.csv"
DEFAULT_CRITERIA = ROOT / "data" / "governance" / "FREEZE_CRITERIA.json"
LEGACY_METADATA = ROOT / "data" / "processed" / "all_pages_metadata.jsonl"
LEGACY_BENCHMARK = ROOT / "data" / "benchmark"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark_governed_v0_1"
CURATED_METADATA = ROOT / "data" / "curated" / "all_pages_metadata.jsonl"
CURATED_PAGES = ROOT / "data" / "curated" / "pages"

TRUE_VALUES = {"1", "true", "yes", "y"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def load_registry(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "doc_id",
        "local_pdf_path",
        "include",
        "language",
        "domain",
        "source_id",
        "template_cluster_id",
        "license_status",
        "redistribution_allowed",
        "duplicate_of",
        "split",
        "page_scope",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Registry missing columns: {sorted(missing)}")
    seen: set[str] = set()
    for row in rows:
        doc_id = row["doc_id"].strip()
        if not doc_id or doc_id in seen:
            raise ValueError(f"Blank or duplicate doc_id in registry: {doc_id!r}")
        seen.add(doc_id)
        row["doc_id"] = doc_id
    return rows


def load_legacy_queries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        for item in read_jsonl(LEGACY_BENCHMARK / split / "queries.jsonl"):
            item = dict(item)
            item.setdefault("metadata", {})
            item["metadata"] = dict(item["metadata"])
            item["metadata"]["legacy_split"] = split
            rows.append(item)
    return rows


def process_missing_documents(
    registry: Sequence[Dict[str, str]],
    metadata_by_doc: Dict[str, List[Dict[str, Any]]],
) -> None:
    CURATED_PAGES.mkdir(parents=True, exist_ok=True)
    processor = PDFProcessor(output_image_dir=CURATED_PAGES, target_dpi=150)
    for row in registry:
        if row["include"].strip().lower() not in TRUE_VALUES:
            continue
        doc_id = row["doc_id"]
        if metadata_by_doc.get(doc_id):
            continue
        pdf_path = ROOT / row["local_pdf_path"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Registered PDF is missing: {pdf_path}")
        domain = DomainType(row["domain"])
        max_pages = None if row["page_scope"] == "full" else 10
        pages = processor.process_pdf(
            pdf_path=pdf_path,
            doc_id=doc_id,
            domain=domain,
            max_pages=max_pages,
        )
        metadata_by_doc[doc_id] = [page.to_dict() for page in pages]


def apply_registry_overrides(
    registry: Sequence[Dict[str, str]],
    metadata_by_doc: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    included_rows = {
        row["doc_id"]: row
        for row in registry
        if row["include"].strip().lower() in TRUE_VALUES
    }
    curated: List[Dict[str, Any]] = []
    for doc_id, row in included_rows.items():
        pages = metadata_by_doc.get(doc_id, [])
        if not pages:
            raise ValueError(f"No processed pages found for included document {doc_id}")
        for page in pages:
            updated = dict(page)
            updated["domain"] = row["domain"]
            curated.append(updated)
    curated.sort(key=lambda item: (item["doc_id"], int(item["page_num"])))
    return curated, included_rows


def find_exact_duplicate_documents(
    curated: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hashes: Dict[str, set[str]] = defaultdict(set)
    for page in curated:
        hashes[page["sha256"]].add(page["doc_id"])
    return [
        {"sha256": digest, "doc_ids": sorted(doc_ids)}
        for digest, doc_ids in hashes.items()
        if len(doc_ids) > 1
    ]


def find_group_leakage(
    included_rows: Mapping[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for field in ("source_id", "template_cluster_id"):
        groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for doc_id, row in included_rows.items():
            value = row[field].strip()
            if value:
                groups[value][row["split"]].append(doc_id)
        for group_id, split_docs in groups.items():
            if len(split_docs) > 1:
                issues.append(
                    {
                        "group_type": field,
                        "group_id": group_id,
                        "splits": {key: sorted(value) for key, value in split_docs.items()},
                    }
                )
    return issues


def build_candidate_queries(
    legacy_queries: Sequence[Dict[str, Any]],
    included_rows: Mapping[str, Dict[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    seen_ids: set[str] = set()
    for query in legacy_queries:
        if query["query_id"] in seen_ids:
            continue
        target_ids = query.get("target_page_ids", [])
        if not target_ids:
            continue
        doc_id = target_ids[0].rsplit("_p", 1)[0]
        row = included_rows.get(doc_id)
        if row is None:
            continue
        candidate = dict(query)
        candidate["domain"] = row["domain"]
        candidate["metadata"] = dict(candidate.get("metadata", {}))
        candidate["metadata"].update(
            {
                "governance_status": "pending_human_validation",
                "candidate_label_only": True,
                "assigned_split": row["split"],
            }
        )
        output[row["split"]].append(candidate)
        seen_ids.add(candidate["query_id"])
    for split in output:
        output[split].sort(key=lambda item: item["query_id"])
    return output


def generate_queries_for_new_documents(
    curated: Sequence[Dict[str, Any]],
    included_rows: Mapping[str, Dict[str, str]],
    legacy_queries: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate heuristic queries for included documents that lack legacy queries."""
    legacy_doc_ids = set()
    for query in legacy_queries:
        target_ids = query.get("target_page_ids", [])
        if target_ids:
            doc_id = target_ids[0].rsplit("_p", 1)[0]
            legacy_doc_ids.add(doc_id)
    
    generator = QueryGenerator(sanitizer=QuerySanitizer())
    new_queries: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    seen_ids = {q["query_id"] for q in legacy_queries}
    
    for doc_id, row in included_rows.items():
        if doc_id in legacy_doc_ids:
            continue
        if row["include"].strip().lower() not in TRUE_VALUES:
            continue
        
        pages = [p for p in curated if p["doc_id"] == doc_id]
        if not pages:
            continue
        
        domain = DomainType(row["domain"])
        split = row["split"]
        prefix = f"q_{split}_{doc_id}"
        
        for page in pages:
            native_text = page.get("native_text", "")
            if not native_text or len(native_text.strip()) < 30:
                continue
            
            queries = generator.generate_queries_for_page(
                domain=domain,
                page_text=native_text,
                page_num=page["page_num"],
                doc_id=doc_id,
                target_page_id=page["page_id"],
                query_id_prefix=prefix,
            )
            
            for q in queries:
                if q.query_id in seen_ids:
                    continue
                q_dict = q.to_dict()
                q_dict["metadata"].update({
                    "governance_status": "pending_human_validation",
                    "candidate_label_only": True,
                    "assigned_split": split,
                })
                new_queries[split].append(q_dict)
                seen_ids.add(q.query_id)
    
    for split in new_queries:
        new_queries[split].sort(key=lambda item: item["query_id"])
    
    return new_queries


def write_annotation_template(
    path: Path,
    queries: Sequence[Dict[str, Any]],
) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for query in queries:
            for page_id in query.get("target_page_ids", []):
                writer.writerow(
                    {
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
                    }
                )


def annotation_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "minimum_judgments_per_pair": 0}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts: Counter[Tuple[str, str]] = Counter()
    valid_rows = 0
    for row in rows:
        if row.get("relevance") in {"0", "1", "2"} and row.get("annotator_id"):
            counts[(row["query_id"], row["page_id"])] += 1
            valid_rows += 1
    return {
        "exists": True,
        "rows": valid_rows,
        "minimum_judgments_per_pair": min(counts.values()) if counts else 0,
        "judged_pairs": len(counts),
    }


def build_stats(
    curated: Sequence[Dict[str, Any]],
    included_rows: Mapping[str, Dict[str, str]],
    queries_by_split: Mapping[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        docs = {doc_id for doc_id, row in included_rows.items() if row["split"] == split}
        pages = [page for page in curated if page["doc_id"] in docs]
        queries = list(queries_by_split[split])
        target_pages = {
            target
            for query in queries
            for target in query.get("target_page_ids", [])
        }
        page_map = {page["page_id"]: page for page in pages}
        target_meta = [page_map[target] for target in target_pages if target in page_map]
        stats[split] = {
            "documents": len(docs),
            "pages": len(pages),
            "queries": len(queries),
            "domains_pages": dict(sorted(Counter(page["domain"] for page in pages).items())),
            "domains_queries": dict(sorted(Counter(query["domain"] for query in queries).items())),
            "source_types_pages": dict(sorted(Counter(page["source_type"] for page in pages).items())),
            "page_types_pages": dict(sorted(Counter(page["page_type"] for page in pages).items())),
            "target_source_types": dict(sorted(Counter(page["source_type"] for page in target_meta).items())),
            "target_page_types": dict(sorted(Counter(page["page_type"] for page in target_meta).items())),
            "query_origins": dict(sorted(Counter(query.get("source", "unknown") for query in queries).items())),
            "source_groups": sorted({included_rows[doc_id]["source_id"] for doc_id in docs}),
        }
    return stats


def evaluate_freeze_gates(
    criteria: Mapping[str, Any],
    stats: Mapping[str, Any],
    included_rows: Mapping[str, Dict[str, str]],
    duplicates: Sequence[Dict[str, Any]],
    group_leakage: Sequence[Dict[str, Any]],
    annotations: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    add("zero_exact_duplicates", not duplicates, list(duplicates))
    add("zero_group_leakage", not group_leakage, list(group_leakage))

    test_rows = [row for row in included_rows.values() if row["split"] == "test"]
    uncleared = [
        row["doc_id"]
        for row in test_rows
        if row["license_status"] != "verified"
        or row["redistribution_allowed"].lower() not in TRUE_VALUES
    ]
    add("test_licenses_verified", not uncleared, sorted(uncleared))

    incomplete = [row["doc_id"] for row in test_rows if row["page_scope"] != "full"]
    add("complete_test_documents", not incomplete, sorted(incomplete))

    domain_pages = stats["test"]["domains_pages"]
    min_pages = int(criteria["min_test_pages_per_domain"])
    domain_coverage = {
        domain: int(domain_pages.get(domain, 0))
        for domain in criteria["required_test_domains"]
    }
    add(
        "test_domain_page_coverage",
        all(value >= min_pages for value in domain_coverage.values()),
        {"minimum": min_pages, "actual": domain_coverage},
    )

    source_counts: Dict[str, set[str]] = defaultdict(set)
    for row in test_rows:
        source_counts[row["domain"]].add(row["source_id"])
    source_actual = {
        domain: len(source_counts.get(domain, set()))
        for domain in criteria["required_test_domains"]
    }
    min_sources = int(criteria["min_test_source_groups_per_domain"])
    add(
        "test_source_diversity",
        all(value >= min_sources for value in source_actual.values()),
        {"minimum": min_sources, "actual": source_actual},
    )

    query_count = int(stats["test"]["queries"])
    add(
        "minimum_test_queries",
        query_count >= int(criteria["min_test_queries"]),
        {"minimum": criteria["min_test_queries"], "actual": query_count},
    )

    origins = stats["test"]["query_origins"]
    human = int(origins.get("human_written", 0))
    human_ratio = human / query_count if query_count else 0.0
    add(
        "human_written_ratio",
        human_ratio >= float(criteria["min_human_written_ratio"]),
        {"minimum": criteria["min_human_written_ratio"], "actual": round(human_ratio, 4)},
    )

    scanned_targets = int(stats["test"]["target_source_types"].get("scanned", 0))
    add(
        "scanned_target_queries",
        scanned_targets >= int(criteria["min_test_scanned_target_queries"]),
        {"minimum": criteria["min_test_scanned_target_queries"], "actual": scanned_targets},
    )

    target_types = stats["test"]["target_page_types"]
    type_coverage = {
        page_type: int(target_types.get(page_type, 0))
        for page_type in criteria["required_query_page_types"]
    }
    min_type_queries = int(criteria["min_test_queries_per_required_page_type"])
    add(
        "target_page_type_coverage",
        all(value >= min_type_queries for value in type_coverage.values()),
        {"minimum": min_type_queries, "actual": type_coverage},
    )

    required_judgments = int(criteria["required_independent_judgments_per_pair"])
    add(
        "independent_human_judgments",
        bool(annotations.get("exists"))
        and int(annotations.get("minimum_judgments_per_pair", 0)) >= required_judgments,
        {"required": required_judgments, **dict(annotations)},
    )
    return gates


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Governed benchmark audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Freeze status: **{report['freeze_status']}**",
        "",
        "## Split statistics",
        "",
        "| Split | Documents | Pages | Queries | Page domains | Query domains |",
        "|---|---:|---:|---:|---|---|",
    ]
    for split in ("train", "dev", "test"):
        stats = report["stats"][split]
        lines.append(
            f"| {split} | {stats['documents']} | {stats['pages']} | {stats['queries']} | "
            f"`{json.dumps(stats['domains_pages'], ensure_ascii=False)}` | "
            f"`{json.dumps(stats['domains_queries'], ensure_ascii=False)}` |"
        )
    lines.extend(["", "## Freeze gates", "", "| Gate | Status | Detail |", "|---|---|---|"])
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "BLOCK"
        detail = json.dumps(gate["detail"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {gate['gate']} | **{status}** | `{detail}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This output is an annotation candidate, not a paper-ready benchmark. "
            "Automatically generated target pages are not final qrels. Resolve every "
            "blocking gate and run the builder with `--freeze` before reporting final test results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze", action="store_true", help="Create an immutable freeze manifest only if all gates pass")
    parser.add_argument("--reset-output", action="store_true", help="Rebuild generated candidate output from scratch")
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    criteria_path = args.criteria.resolve()
    output = args.output.resolve()
    if args.reset_output and output.exists():
        if ROOT not in output.parents or output == ROOT:
            raise ValueError(f"Refusing to remove unsafe output path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    registry = load_registry(registry_path)
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))

    metadata_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source_path in (CURATED_METADATA, LEGACY_METADATA):
        if source_path.exists():
            for page in read_jsonl(source_path):
                key = (page["doc_id"], page["page_num"])
                existing = next((p for p in metadata_by_doc[page["doc_id"]] if p["page_num"] == page["page_num"]), None)
                if existing is None:
                    metadata_by_doc[page["doc_id"]].append(page)
    process_missing_documents(registry, metadata_by_doc)
    curated, included_rows = apply_registry_overrides(registry, metadata_by_doc)
    write_jsonl(CURATED_METADATA, curated)

    duplicates = find_exact_duplicate_documents(curated)
    group_leakage = find_group_leakage(included_rows)
    legacy_queries = load_legacy_queries()
    queries_by_split = build_candidate_queries(legacy_queries, included_rows)
    new_queries = generate_queries_for_new_documents(curated, included_rows, legacy_queries)
    for split in ("train", "dev", "test"):
        queries_by_split[split].extend(new_queries[split])
        queries_by_split[split].sort(key=lambda item: item["query_id"])

    for split in ("train", "dev", "test"):
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_docs = {doc_id for doc_id, row in included_rows.items() if row["split"] == split}
        split_pages = [page for page in curated if page["doc_id"] in split_docs]
        write_jsonl(split_dir / "queries_candidates.jsonl", queries_by_split[split])
        write_jsonl(split_dir / "pages_metadata.jsonl", split_pages)
        (split_dir / "corpus_pages.json").write_text(
            json.dumps([page["page_id"] for page in split_pages], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_annotation_template(split_dir / "annotations_template.tsv", queries_by_split[split])

    annotations = annotation_summary(output / "test" / "annotations_final.tsv")
    stats = build_stats(curated, included_rows, queries_by_split)
    gates = evaluate_freeze_gates(
        criteria, stats, included_rows, duplicates, group_leakage, annotations
    )
    freeze_status = "READY" if all(gate["passed"] for gate in gates) else "BLOCKED"
    report = {
        "benchmark_version": criteria["benchmark_version"],
        "generated_at": utc_now(),
        "freeze_status": freeze_status,
        "registry_sha256": sha256_file(registry_path),
        "criteria_sha256": sha256_file(criteria_path),
        "curated_metadata_sha256": sha256_file(CURATED_METADATA),
        "stats": stats,
        "exact_duplicate_documents": duplicates,
        "group_leakage": group_leakage,
        "annotations": annotations,
        "gates": gates,
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown_report(report)
    (output / "audit_report.md").write_text(markdown, encoding="utf-8")

    split_assignment = {
        split: sorted(doc_id for doc_id, row in included_rows.items() if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    candidate_lock = {
        "status": "candidate_assignment_locked",
        "created_at": report["generated_at"],
        "registry_sha256": report["registry_sha256"],
        "assignment": split_assignment,
        "assignment_sha256": sha256_json(split_assignment),
        "warning": "Document assignment is locked for annotation, but test qrels are not frozen.",
    }
    (output / "candidate_split_lock.json").write_text(
        json.dumps(candidate_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    blocked_path = output / "FREEZE_BLOCKED.md"
    frozen_path = output / "FROZEN_MANIFEST.json"
    if freeze_status == "BLOCKED":
        blocked_path.write_text(markdown, encoding="utf-8")
        if frozen_path.exists():
            frozen_path.unlink()
    elif blocked_path.exists():
        blocked_path.unlink()

    if args.freeze:
        if freeze_status != "READY":
            print(f"[BLOCKED] Freeze gates failed. See {blocked_path}")
            return 2
        manifest = {
            "benchmark_version": report["benchmark_version"],
            "frozen_at": utc_now(),
            "registry_sha256": report["registry_sha256"],
            "criteria_sha256": report["criteria_sha256"],
            "curated_metadata_sha256": report["curated_metadata_sha256"],
            "candidate_assignment_sha256": candidate_lock["assignment_sha256"],
            "test_queries_sha256": sha256_file(output / "test" / "queries_candidates.jsonl"),
            "test_annotations_sha256": sha256_file(output / "test" / "annotations_final.tsv"),
            "gates": gates,
        }
        frozen_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FROZEN] Wrote {frozen_path}")
    else:
        print(f"[{freeze_status}] Wrote governed candidate to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
