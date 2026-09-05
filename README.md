# Vi-ViDoRe: A Governed Multi-Domain Benchmark for Vietnamese Visual Document Retrieval

Vi-ViDoRe is a **research-oriented benchmark project** for Vietnamese visual document retrieval. This repository is the working research codebase for building and auditing a governed benchmark, running retrieval baselines, supporting human annotation, and evaluating results with reproducible statistical testing.

This is **not only an application repository**. It is intended to support research on how retrieval systems find evidence pages in real Vietnamese PDF documents across domains such as legal, finance, healthcare, and education.

> **Research status:** the repository is currently an **alpha benchmark candidate**. The codebase is implemented enough for experimentation, but the dataset is **not yet frozen** and must not be presented as a final publication-ready benchmark until governance, annotation, and freeze gates are completed.

## Research motivation

Vietnamese document retrieval is difficult because relevant evidence may appear in scanned pages, OCR-noisy text, tables, charts, forms, and long structured PDFs. Text-only retrieval can struggle in these settings. Vi-ViDoRe is designed to study whether dense and visual multi-vector retrieval methods can better recover evidence pages under realistic document conditions.

## Research objective

Given a Vietnamese query and a governed corpus of PDF pages, a system must rank the pages that contain relevant evidence.

```text
Vietnamese query
      +
Governed PDF collection
      ↓
Text / dense / visual page representations
      ↓
Retrieval and ranking
      ↓
Top-k evidence pages
      ↓
Human-validated qrels and statistical evaluation
```

The project evaluates **retrieval quality**, not answer generation. Its role is closer to an evidence-retrieval benchmark than to a full RAG benchmark.

## Research questions

This repository is structured around questions such as:

1. How much do visual retrievers help on Vietnamese document pages compared with lexical and dense baselines?
2. Which page conditions benefit most from visual retrieval, such as scans, tables, charts, and forms?
3. How should a Vietnamese retrieval benchmark be governed so that legal, scientific, and annotation risks remain auditable?
4. How much performance variance appears across legal, finance, healthcare, and education domains?
5. Which improvements remain statistically reliable after paired significance tests and multiple-comparison correction?

## What the repository currently contains

The current folder state includes:

- benchmark-building scripts in `scripts/`;
- source code in `src/` for data processing, models, training, and evaluation;
- tests in `tests/`;
- governance files in `data/governance/`;
- research-support documents including `DATASET_CARD.md`, `MODEL_CARD.md`, `AI_AUDIT_REPORT.md`, `PUBLICATION_READINESS_REVIEW.md`, and `BENCHMARK_SUBMISSION_CHECKLIST.md`;
- reproducibility files including `pyproject.toml`, `requirements.lock`, `.pre-commit-config.yaml`, `Dockerfile`, and `docker-compose.yml`.

The repository does **not** currently include raw source PDFs, a frozen benchmark release, or final paper-ready leaderboard results.

## Current benchmark status

| Component | Current state |
|---|---|
| Research codebase | Implemented |
| Governance workflow | Implemented |
| Annotation tooling | Implemented |
| Baseline evaluation pipeline | Implemented |
| Statistical testing | Implemented |
| Test suite | Present |
| Human-reviewed final dataset | Incomplete |
| Frozen benchmark manifest | Not available yet |
| Publication-ready results | Not available yet |

## Core research contributions

1. A governed Vietnamese visual document retrieval benchmark pipeline.
2. Document-level provenance, licensing, and freeze-gate checks.
3. Support for BM25, dense bi-encoders, hybrid retrieval, and visual multi-vector retrievers.
4. A human annotation workflow with pooling, double judgment, and adjudication.
5. A reproducible evaluation stack with confidence intervals, paired tests, and multiple-comparison correction.

## Main workflows

```bash
python scripts/validate_registry.py
python scripts/verify_licenses.py
python scripts/contamination_report.py
python scripts/05_build_governed_benchmark.py
python scripts/05_build_governed_benchmark.py --freeze
python scripts/create_pilot100_template.py
python scripts/pool_candidates.py
python scripts/adjudicate.py ann_A.tsv ann_B.tsv qrels_final.tsv report.json
python scripts/03_run_baselines.py --split dev
python scripts/03_run_baselines.py --split test --debug
python scripts/04_train_adaptation.py
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/Ducduyzs/Vi-ViDoRe-A-Governed-Multi-Domain-Benchmark-for-Vietnamese-Visual-Document-Retrieval.git
cd Vi-ViDoRe-A-Governed-Multi-Domain-Benchmark-for-Vietnamese-Visual-Document-Retrieval

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Important documents

- `DATASET_CARD.md`
- `MODEL_CARD.md`
- `ANNOTATION_GUIDELINE_v1.0.md`
- `PUBLICATION_READINESS_REVIEW.md`
- `BENCHMARK_SUBMISSION_CHECKLIST.md`
- `CHANGELOG.md`

## License

Code is released under the `MIT License`. Document licenses remain source-specific and must be tracked through the governance workflow.

## Citation

```bibtex
@misc{vi_vidore_2026,
  title  = {Vi-ViDoRe: A Governed Multi-Domain Benchmark for Vietnamese Visual Document Retrieval},
  author = {Vi-ViDoRe Contributors},
  year   = {2026},
      note   = {Alpha benchmark candidate; dataset not yet frozen}
}
```
