import csv
import importlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILDER = importlib.import_module("scripts.05_build_governed_benchmark")


def _registry_rows():
    with (ROOT / "data/governance/document_registry.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def test_registry_excludes_non_vietnamese_and_known_duplicate():
    rows = _registry_rows()
    included = [row for row in rows if row["include"] == "true"]
    assert included
    assert all(row["language"] == "vi" for row in included)

    duplicate = next(row for row in rows if row["doc_id"] == "1409.1556v6 (1)")
    assert duplicate["include"] == "false"
    assert duplicate["duplicate_of"] == "1409.1556v6"


def test_registry_source_and_template_groups_do_not_cross_splits():
    rows = [row for row in _registry_rows() if row["include"] == "true"]
    for field in ("source_id", "template_cluster_id"):
        groups = defaultdict(set)
        for row in rows:
            groups[row[field]].add(row["split"])
        assert all(len(splits) == 1 for splits in groups.values())


def test_governed_corpus_has_unique_page_ids_and_required_domains():
    metadata_path = ROOT / "data/curated/all_pages_metadata.jsonl"
    pages = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    page_ids = [page["page_id"] for page in pages]
    assert len(page_ids) == len(set(page_ids))
    assert {"education", "legal", "financial", "healthcare"}.issubset(
        {page["domain"] for page in pages}
    )


def test_current_candidate_is_explicitly_blocked_from_freeze():
    report = json.loads(
        (ROOT / "data/benchmark_governed_v0_1/audit_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["freeze_status"] == "BLOCKED"
    assert not (ROOT / "data/benchmark_governed_v0_1/FROZEN_MANIFEST.json").exists()
    blocked = {gate["gate"] for gate in report["gates"] if not gate["passed"]}
    assert "test_licenses_verified" in blocked
    assert "independent_human_judgments" in blocked
    assert "test_licenses_verified" in blocked
    assert "complete_test_documents" in blocked
    assert "test_source_diversity" in blocked
    assert "human_written_ratio" in blocked
    assert "scanned_target_queries" in blocked
    assert "target_page_type_coverage" in blocked


def test_exact_duplicate_detector_operates_at_document_level():
    pages = [
        {"doc_id": "doc_a", "sha256": "same"},
        {"doc_id": "doc_a", "sha256": "same"},
        {"doc_id": "doc_b", "sha256": "same"},
    ]
    duplicates = BUILDER.find_exact_duplicate_documents(pages)
    assert duplicates == [{"sha256": "same", "doc_ids": ["doc_a", "doc_b"]}]

