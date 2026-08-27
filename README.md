# Vi-ViDoRe: A Governed Multi-Domain Benchmark for Vietnamese Visual Document Retrieval

Vi-ViDoRe is a governed benchmark for evaluating text, dense, hybrid, and
vision-language retrieval over real Vietnamese PDF documents. It focuses on
finding the correct evidence page for a Vietnamese query across legal,
financial, healthcare, and education domains.

> **Status:** benchmark candidate — infrastructure is implemented, but the
> dataset is not frozen or publication-ready until the remaining legal review,
> human query authoring, and double-annotation requirements are completed.

## Research scope

Given a Vietnamese query and a collection of PDF pages, a system must rank the
pages that contain relevant evidence.

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

Vi-ViDoRe evaluates the retrieval stage. It is not currently presented as a
complete answer-generation or RAG benchmark.

## Why Vi-ViDoRe

Vietnamese document retrieval is difficult because evidence can appear in:

- born-digital PDF text;
- scanned pages with imperfect OCR;
- tables and financial statements;
- charts and figures;
- forms and administrative layouts;
- long, domain-specific documents.

The benchmark combines realistic document conditions with explicit data
governance so that benchmark results can be reproduced and legally audited.

## Core contributions

1. A multi-domain Vietnamese visual document retrieval benchmark.
2. Document-level governance with source, license, page-scope, and checksum
   tracking.
3. Human-authored queries and independently judged page-level qrels.
4. Comparable lexical, dense, hybrid, and visual retrieval baselines.
5. Reproducible evaluation with paired tests, confidence intervals, and
   Holm-Bonferroni correction.
6. Subgroup analysis for domains, scans, tables, charts, forms, and document
   types.

## Current readiness

| Component | Status |
|---|---|
| Packaging, tests, Docker, and dependency lock | Complete |
| Governance and contamination tooling | Complete |
| Baseline and significance-testing harness | Complete |
| Annotation, pooling, and adjudication tooling | Complete |
| PDF license and page-scope review | Human/legal action required |
| Domain-source diversity | Additional sources required |
| 355 expert-reviewed Vietnamese queries | In progress |
| Scan, table, chart, and form coverage | Human curation required |
| Double annotation and adjudication | Not complete |
| Frozen benchmark and final test results | Not complete |

The current benchmark candidate and any pilot scores are for pipeline debugging.
They must not be reported as final paper results.

## Retrieval systems

The governed runner supports or is designed to compare:

- BM25 over native text or OCR;
- Vietnamese and multilingual dense retrievers;
- hybrid lexical+dense retrieval;
- ColPali/ColQwen-style visual multi-vector retrieval;
- reciprocal-rank fusion over heterogeneous retrievers;
- optional reranking and adaptation experiments.

Mandatory baselines fail fast instead of silently recording missing runs as
zero-valued results.

## Data governance

Each document is registered with:

- stable document ID;
- source organization and URL;
- license and redistribution status;
- approved page scope;
- SHA-256 checksum;
- domain and source type;
- OCR/layout metadata;
- split assignment.

Governance checks cover duplicate PDFs, near-duplicate documents, split leakage,
license completeness, source diversity, page-type balance, human-query ratios,
and independent judgments.

Raw PDFs are intentionally not distributed in this repository. Users must
obtain documents from approved sources and follow their licenses.

## Annotation workflow

Candidate pages are pooled from BM25, dense, and visual retrievers using
reciprocal-rank fusion. Two annotators independently judge each query-page pair,
after which disagreements are adjudicated into final qrels.

```text
BM25 ──────────────┐
Dense Vietnamese ──┤
Multilingual dense ┼→ RRF pool → two annotators → adjudication → frozen qrels
ColPali / ColQwen ─┘
```

See [ANNOTATION_GUIDELINE_v1.0.md](ANNOTATION_GUIDELINE_v1.0.md) and
[data/governance/DATA_GOVERNANCE_POLICY.md](data/governance/DATA_GOVERNANCE_POLICY.md).

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

Dependencies are recorded in `requirements.lock`. API credentials, local
configuration, office documents, and raw PDFs are excluded from Git.

## Governance commands

```bash
python scripts/validate_registry.py
python scripts/verify_licenses.py
python scripts/contamination_report.py
```

All blocking findings must be resolved before freezing the benchmark.

## Annotation pilot

```bash
python scripts/create_pilot100_template.py
python scripts/pilot_tracker.py assign annotator_A 25
python scripts/adjudicate.py ann_A.tsv ann_B.tsv qrels_final.tsv report.json
```

The pilot is used to refine the guideline and measure inter-annotator
agreement before scaling annotation.

## Freeze the benchmark

```bash
python scripts/05_build_governed_benchmark.py --freeze
```

Freezing is permitted only when the governance gates pass. The frozen release
must record dataset version, manifest hashes, qrels hash, random seeds, model
revisions, annotation-guideline version, and contamination report.

## Run governed baselines

Development experiments should use the dev split. The test split should be run
only after models and thresholds are locked.

```bash
python scripts/03_run_baselines.py --split dev
python scripts/03_run_baselines.py --split test
```

Reports include runtime metadata and subgroup results. Missing subgroups are
reported as `N/A (n=0)`, not as `0.0`.

## Evaluation

Primary retrieval metrics include:

- Recall@k and Hit Rate@k;
- MRR;
- nDCG@k;
- macro results across domains;
- paired confidence intervals and randomization tests;
- Holm-Bonferroni-corrected comparisons;
- latency, memory, and index-size measurements.

Results are also decomposed by domain, source type, scan status, page type, and
query difficulty.

## Repository layout

```text
src/data/           PDF processing, schema, deduplication, query sanitation
src/models/         BM25, dense, visual retrieval, and MaxSim
src/evaluation/     Metrics, significance testing, and report generation
src/training/       Adaptation and hard-negative mining
scripts/            Governance, annotation, benchmark, and experiment commands
tests/              Unit and integration tests
data/governance/    Registry, policies, and freeze criteria
```

## Reproducibility and responsible release

Before reporting final results:

1. resolve every license and page-scope blocker;
2. complete expert query review;
3. double-annotate and adjudicate dev/test qrels;
4. freeze manifests before final experiments;
5. lock code, configuration, model revisions, and seeds;
6. run the test split once under the frozen protocol;
7. publish confidence intervals, corrected significance tests, and failure
   analysis.

See [BENCHMARK_SUBMISSION_CHECKLIST.md](BENCHMARK_SUBMISSION_CHECKLIST.md),
[DATASET_CARD.md](DATASET_CARD.md), and [MODEL_CARD.md](MODEL_CARD.md).

## License

Code is released under the [MIT License](LICENSE). Document licenses remain
source-specific and are tracked in the governance registry. Benchmark
annotations must not be released until their legal and human-validation gates
have passed.

## Citation

```bibtex
@misc{vi_vidore_2026,
  title  = {Vi-ViDoRe: A Governed Multi-Domain Benchmark for Vietnamese Visual Document Retrieval},
  author = {Vi-ViDoRe Contributors},
  year   = {2026},
  note   = {Benchmark candidate; not yet frozen}
}
```
