"""Idempotent and stricter entry point for the governed benchmark builder."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def main() -> int:
    builder = importlib.import_module("scripts.05_build_governed_benchmark")

    # Once curated metadata exists it is authoritative. Re-appending legacy
    # metadata would duplicate every previously curated page on repeated runs.
    if builder.CURATED_METADATA.exists():
        builder.LEGACY_METADATA = Path("__governance_no_legacy_metadata__")

    original_evaluator = builder.evaluate_freeze_gates

    def evaluate_with_domain_query_gate(
        criteria: Mapping[str, Any],
        stats: Mapping[str, Any],
        included_rows: Mapping[str, Dict[str, str]],
        duplicates: Sequence[Dict[str, Any]],
        group_leakage: Sequence[Dict[str, Any]],
        annotations: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        gates = original_evaluator(
            criteria,
            stats,
            included_rows,
            duplicates,
            group_leakage,
            annotations,
        )
        minimum = int(criteria.get("min_test_queries_per_domain", 0))
        actual = {
            domain: int(stats["test"]["domains_queries"].get(domain, 0))
            for domain in criteria["required_test_domains"]
        }
        gates.append(
            {
                "gate": "test_query_domain_coverage",
                "passed": all(value >= minimum for value in actual.values()),
                "detail": {"minimum": minimum, "actual": actual},
            }
        )
        return gates

    builder.evaluate_freeze_gates = evaluate_with_domain_query_gate
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())

