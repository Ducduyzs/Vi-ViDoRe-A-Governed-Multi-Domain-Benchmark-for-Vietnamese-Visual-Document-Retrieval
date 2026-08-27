"""Idempotent entry point for the governed benchmark builder.

Use this wrapper instead of invoking ``05_build_governed_benchmark.py`` by
path. When curated metadata already exists, it is authoritative and the legacy
metadata is not concatenated a second time.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def main() -> int:
    builder = importlib.import_module("scripts.05_build_governed_benchmark")
    if builder.CURATED_METADATA.exists():
        builder.LEGACY_METADATA = Path("__governance_no_legacy_metadata__")
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())

