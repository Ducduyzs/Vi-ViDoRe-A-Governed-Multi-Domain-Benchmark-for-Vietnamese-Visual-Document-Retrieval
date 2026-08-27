# Vi-ViDoRe Benchmark Submission Checklist

**Version**: 1.0  
**Target**: ACL / EMNLP / SIGIR / NAACL / COLING / LREC-COLING  
**Status**: Candidate (freeze gates not yet passed)

---

## 📋 Phase 1: Data Governance (MUST PASS before freeze)

### Freeze Gates (from `FREEZE_CRITERIA.json`)

| Gate | Criteria | Current | Required | Status |
|------|----------|---------|----------|--------|
| `zero_exact_duplicates` | No SHA-256 duplicates | 0 | 0 | ✅ PASS |
| `zero_group_leakage` | No source/template cross-split | 0 | 0 | ✅ PASS |
| `test_licenses_verified` | All test docs verified | 0/6 | 6/6 | ❌ BLOCK |
| `complete_test_documents` | All test docs full scope | 0/6 | 6/6 | ❌ BLOCK |
| `test_domain_page_coverage` | ≥10 pages/domain | 30/20/51/198 | 10 each | ✅ PASS |
| `test_source_diversity` | ≥2 sources/domain | 3/1/1/2 | 2 each | ❌ BLOCK |
| `minimum_test_queries` | ≥500 queries | 888 | 500 | ✅ PASS |
| `human_written_ratio` | ≥40% human | 0% | 40% | ❌ BLOCK |
| `scanned_target_queries` | ≥20 scanned | 0 | 20 | ❌ BLOCK |
| `target_page_type_coverage` | ≥20 each type | 4/2/0 | 20 each | ❌ BLOCK |
| `independent_human_judgments` | 2 annotations/pair | 0 | 2 | ❌ BLOCK |
| `test_query_domain_coverage` | ≥50 queries/domain | 11/163/699/15 | 50 each | ❌ BLOCK |

### Required Actions

- [ ] **Legal review 6 test PDFs**: Verify redistribution rights, update registry
- [ ] **Download full PDFs** for 6 test docs (or replace with full-scope alternatives)
- [ ] **Add legal source #2** (e.g., Bộ Tư pháp, Luật Việt Nam, different university)
- [ ] **Add financial source #2** (e.g., IMF, ADB, VN bank annual report)
- [ ] **Write 355 human queries** (≥40% of 888)
- [ ] **Create 20+ scanned queries** (target docs 07, 08)
- [ ] **Create 20+ form/table/chart queries** each
- [ ] **Add 39 education + 35 legal queries**
- [ ] **Run pilot 100 double annotation** (Cohen's κ ≥ 0.67)
- [ ] **Scale annotation to full test set** (888 queries)

### Freeze Command

```bash
# After all gates pass
$env:PYTHONPATH="."; python scripts/05_build_governed_benchmark.py --freeze
# Creates: data/benchmark_governed_v0_1/FROZEN_MANIFEST.json
```

---

## 📊 Phase 2: Baseline Suite (on FROZEN test)

### Required Baselines

| Baseline | Implementation | Status |
|----------|----------------|--------|
| BM25 (native text) | `src/models/text_baselines.py` | ✅ Done |
| BM25 (OCR text) | Need OCR pipeline | ❌ TODO |
| Dense Bi-Encoder (mBERT) | `DenseBiEncoderRetriever` | ✅ Done |
| Dense Bi-Encoder (BGE-M3) | `DenseBiEncoderRetriever` | ✅ Done |
| Dense Bi-Encoder (Vietnamese) | `bkai-foundation-models/vietnamese-bi-encoder` | ✅ Done |
| ColPali v1.2 (zero-shot) | `ColPaliVisualRetriever` | ✅ Done |
| ColQwen2 (zero-shot) | `ColPaliVisualRetriever` | ⚠️ Need test |
| ColPali fine-tuned (Vi-ViDoRe) | `scripts/04_train_adaptation.py` | ❌ Not trained |
| Full-document / Long-context | Need implementation | ❌ TODO |

### Ablation Studies (for model paper)

| Ablation | Status |
|----------|--------|
| LoRA rank (8, 16, 32, 64) | ❌ |
| Hard negative strategy | ❌ |
| In-PDF negative weight | ❌ |
| Temperature scaling | ❌ |
| Curriculum learning | ❌ |
| No adapter (zero-shot) | ✅ |
| Visual only vs Text only | ❌ |

### Cross-Domain Validation

- [ ] Train on Edu, test on Legal/Fin/Health
- [ ] Train on Legal, test on Edu/Fin/Health
- [ ] Train on Fin, test on Edu/Legal/Health
- [ ] Train on Health, test on Edu/Legal/Fin

---

## 📈 Phase 3: Statistical Rigor

### Significance Testing

- [ ] **Paired bootstrap** (10,000 iterations) for all pairwise comparisons
- [ ] **Randomization test** as secondary
- [ ] **Holm-Bonferroni** correction for multiple baselines
- [ ] **Effect sizes** (Cohen's d) reported
- [ ] **Confidence intervals** (95%, bootstrap) for all metrics
- [ ] **Per-query** differences analyzed

### Reporting Standards

- [ ] Macro-domain nDCG@5 (primary)
- [ ] Overall nDCG@5 with 95% CI
- [ ] Per-domain nDCG@5 with sample counts
- [ ] Per-source-type (born-digital vs scanned)
- [ ] Per-page-type (text/table/chart/form)
- [ ] Per-query-type (fact_lookup, legal_clause, etc.)
- [ ] MRR@10, Recall@5, Recall@10
- [ ] Latency: P50, P95 (ms)
- [ ] Index size, VRAM, throughput
- [ ] Hardware: GPU, CUDA, PyTorch versions

---

## 🔬 Phase 4: Error Analysis (Required for Publication)

### Qualitative Analysis

- [ ] **Failure cases** by domain
- [ ] **Failure cases** by page type
- [ ] **Failure cases** by query type
- [ ] **False positives** analysis
- [ ] **False negatives** analysis
- [ ] **Scanned vs born-digital** gap analysis
- [ ] **Attribution quality** (if fine-tuned with citations)

### Human Evaluation (if claiming "human-validated")

- [ ] **100-200 claims** double-annotated for attribution
- [ ] Cohen's κ for attribution labels
- [ ] Citation correctness (entailment)
- [ ] Drift detection (citation → claim alignment)

---

## 📦 Phase 5: Reproducibility Package

### Code & Environment

- [ ] **Git repository** with immutable commit hash
- [ ] **requirements.lock** (pinned dependencies)
- [ ] **Dockerfile** + **docker-compose.yml**
- [ ] **CITATION.cff**
- [ ] **LICENSE** (MIT + data licenses)
- [ ] **CONTRIBUTING.md**
- [ ] **Pre-commit hooks** (ruff, mypy, pytest)

### Artifacts (per run)

- [ ] **Run manifest**: timestamp, seed, config, hardware, commit
- [ ] **Per-query results**: scores, rankings
- [ ] **Model checkpoints**: revision hash, training config
- [ ] **Baseline outputs**: all retrievers on frozen test
- [ ] **Significance test outputs**: p-values, CIs, effect sizes

### Documentation

- [ ] **README.md**: Setup, data status, commands, limitations
- [ ] **DATASET_CARD.md**: HuggingFace format
- [ ] **MODEL_CARD.md**: For fine-tuned models
- [ ] **ANNOTATION_GUIDELINE_v1.0.md**
- [ ] **CONTAMINATION_REPORT.md**
- [ ] **CHANGELOG.md**

---

## 📝 Phase 6: Paper Preparation

### Resource/Benchmark Paper

| Section | Requirements |
|---------|--------------|
| Abstract | Benchmark scope, key stats, main results |
| Related Work | Vietnamese IR, visual retrieval, benchmarks |
| Data Governance | Registry, licenses, splits, contamination audit |
| Annotation | Guideline, double annotation, κ, adjudication |
| Statistics | Domain balance, query types, page types |
| Baselines | BM25, dense, ColPali, ColQwen, significance |
| Analysis | Error types, domain gaps, scanned gap |
| Ethics | Licenses, bias, privacy, access |
| Reproducibility | Docker, lockfile, seeds, hardware |

### Model Paper (if submitting adaptation)

| Section | Requirements |
|---------|--------------|
| Method | Architecture, LoRA, training objective, negatives |
| Experiments | Vi-ViDoRe test + ablations + cross-domain |
| Baselines | Strong (BM25, BGE-M3, ColPali zero-shot, full-doc) |
| Seeds | ≥3 seeds, mean ± std |
| Retention | Zero-shot ViDoRe performance retained |
| OOD | Cross-domain, cross-backbone validation |
| Efficiency | Latency, VRAM, index size, throughput |
| Ablation | LoRA rank, negatives, temperature, curriculum |

---

## ✅ Final Pre-Submission Checks

### Code Quality

- [ ] `python -m pytest tests/ -v` → all pass
- [ ] `ruff check src/ scripts/` → no errors
- [ ] `mypy src/` → no errors (strict)
- [ ] `black --check src/ scripts/` → formatted

### Data Integrity

- [ ] `python scripts/validate_registry.py` → only annotation gate blocked
- [ ] `python scripts/contamination_report.py` → CLEAN
- [ ] `python scripts/05_build_governed_benchmark.py --freeze` → FROZEN_MANIFEST created

### Experiment Reproduction

- [ ] `python scripts/03_run_baselines.py --split test` → runs without error
- [ ] Results saved to `results/benchmark_test_results.{json,md,tex}`
- [ ] Significance tests run and saved

### Documentation

- [ ] README.md complete (setup, commands, limitations)
- [ ] DATASET_CARD.md valid HuggingFace format
- [ ] MODEL_CARD.md (if model paper)
- [ ] CONTRIBUTING.md
- [ ] LICENSE + CITATION.cff

### Legal/Ethical

- [ ] All test documents license-verified
- [ ] Redistribution rights confirmed
- [ ] Annotations CC BY 4.0
- [ ] No PII in data
- [ ] Bias statement included

---

## 🚀 Submission Commands

```bash
# 1. Final validation
python -m pytest tests/ -v
python scripts/validate_registry.py
python scripts/contamination_report.py

# 2. Freeze benchmark
$env:PYTHONPATH="."; python scripts/05_build_governed_benchmark.py --freeze

# 3. Run full baseline suite on frozen test
python scripts/03_run_baselines.py --split test

# 4. Run significance tests (in notebook or script)
python -c "
from src.evaluation.evaluator import ViViDoReEvaluator
# evaluator.significance_test(...)
"

# 5. Package artifacts
git tag v1.0.0-submission
git push origin v1.0.0-submission

# 6. Build Docker
docker build -t vi-vidore:v1.0.0 .
docker push registry/vi-vidore:v1.0.0
```

---

## 📞 Emergency Contacts

| Issue | Contact |
|-------|---------|
| License/legal | [Legal counsel] |
| Annotation quality | [Annotation lead] |
| Technical/reproducibility | [Tech lead] |
| Paper writing | [PI] |

---

**Last Updated**: 2026-08-27  
**Next Review**: After Phase 1 complete (freeze gates pass)