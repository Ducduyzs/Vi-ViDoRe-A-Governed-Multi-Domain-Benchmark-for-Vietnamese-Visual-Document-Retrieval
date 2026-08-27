# Changelog

All notable changes to Vi-ViDoRe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-27

### Added
- **Governed benchmark architecture** (`scripts/05_build_governed_benchmark.py`):
  - Document registry with provenance, license verification, split locking
  - Freeze gates for data governance (12 gates)
  - Automated audit reports (Markdown + JSON)
  - Candidate split lock with SHA-256 hashing
- **Data governance scripts**:
  - `scripts/verify_licenses.py` - PDF license extraction and verification
  - `scripts/validate_registry.py` - Registry validation against freeze criteria
  - `scripts/contamination_report.py` - Exact/near duplicate detection, source leakage check
- **Annotation pipeline**:
  - `ANNOTATION_GUIDELINE_v1.0.md` - Relevance scale (0/1/2), double annotation protocol
  - `scripts/create_pilot100_template.py` - Balanced pilot 100 query sampling
  - `scripts/adjudicate.py` - Cohen's kappa, majority vote, adjudication report
  - `scripts/pool_candidates.py` - RRF pooling from BM25 + Dense Vi + ColPali/ColQwen
  - `scripts/balance_domains.py` - Human-written query templates, domain balancing
  - `scripts/pilot_tracker.py` - Progress tracking, assignment, kappa monitoring
- **Evaluation infrastructure**:
  - Paired bootstrap test with centered resampling (`compute_paired_bootstrap_test`)
  - Randomization/permutation test (`compute_randomization_test`)
  - Holm-Bonferroni multiple comparison correction
  - Deterministic bootstrap CI with configurable seed
  - Per-query metrics storage for significance testing
  - Evaluator `significance_test()` method for pairwise model comparison
- **Model configuration**:
  - `ModelConfig.backbone_revision` for HF model pinning
  - `--visual_revision` CLI argument for ColPali/ColQwen
- **Baseline evaluation**:
  - Fail-fast mandatory baselines (`run_baseline()` helper)
  - Runtime metadata in reports (timestamp, Python, PyTorch, CUDA, GPU, seed, model revisions)
  - N/A display for missing subgroups instead of 0.0
  - Source type inference via majority vote over target pages
- **Reproducibility**:
  - `pyproject.toml` with complete metadata, entry points, tool configs
  - `requirements.lock` (340 pinned dependencies)
  - `Dockerfile` + `docker-compose.yml` (eval, build, train, Jupyter services)
  - `.pre-commit-config.yaml` (ruff, mypy, bandit, pytest)
  - `CITATION.cff`, `LICENSE` (MIT + data licenses), `CONTRIBUTING.md`
- **Documentation**:
  - `README.md` - Complete setup, data status, commands, limitations, publication readiness
  - `DATASET_CARD.md` - HuggingFace format dataset card
  - `MODEL_CARD.md` - Fine-tuned model documentation template
  - `BENCHMARK_SUBMISSION_CHECKLIST.md` - 6-phase publication checklist
- **Tests** (38 tests):
  - `test_significance_adjudication.py` - 16 tests for bootstrap, randomization, kappa, adjudication, pooling, contamination
  - All existing 22 tests maintained

### Fixed
- **P0/P1 Protocol Issues** (from `PUBLICATION_READINESS_REVIEW.md`):
  - CLI `--run_visual` flag uses `BooleanOptionalAction` for disable support
  - Evaluator shows N/A for missing subgroups instead of 0.0
  - Source type inference uses majority vote over all target pages
  - Bootstrap RNG seed for deterministic CIs (default 42)
  - Query generation logs exceptions instead of swallowing
  - Paired bootstrap + randomization significance tests added
  - Model revision pinning for HuggingFace checkpoints
  - Baseline script fail-fast on mandatory baseline failures
  - Per-query rankings + runtime metadata in JSON reports
- **TOML compliance** - Removed inline comments from `pyproject.toml`
- **Adjudication** - `load_annotations` supports optional annotator filter

### Changed
- Bootstrap CI now uses `np.random.default_rng(seed)` for determinism
- Paired bootstrap test centers differences under H0 for proper null testing
- Source type inference uses majority vote across all target pages
- Report generator shows "N/A (n=0)" for missing subgroups with sample counts
- Baseline script uses `BooleanOptionalAction` for `--run_visual`

### Security
- Bandit security linting in pre-commit hooks
- No hardcoded secrets; API keys via `config.local.json` + env vars

## [0.0.1] - 2026-08-20

### Added
- Initial Vi-ViDoRe prototype
- ColPali visual retriever with MultiVectorEncoder backend
- BM25 and Dense Bi-Encoder baselines
- MaxSim implementation with variable-length query support
- Vietnamese query generation with LLM assistance
- PDF processing pipeline (PyMuPDF + pdf2image)
- Basic evaluation metrics (NDCG, Recall, MRR, bootstrap CI)
- Legacy benchmark generation (`scripts/02_generate_benchmark.py`)