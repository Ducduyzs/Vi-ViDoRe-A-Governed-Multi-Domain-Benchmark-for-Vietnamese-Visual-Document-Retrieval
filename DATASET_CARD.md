---
annotations_creators:
  - expert-generated
  - crowdsourced
language:
  - vi
language_creators:
  - expert-generated
license:
  - cc-by-4.0
  - other
multilinguality:
  - monolingual
size_categories:
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - information-retrieval
  - visual-document-retrieval
task_ids:
  - multi-vector-retrieval
  - late-interaction-retrieval
pretty_name: "Vi-ViDoRe: Vietnamese Visual Document Retrieval Benchmark"
---

# Dataset Card for Vi-ViDoRe

## Table of Contents
- [Dataset Description](#dataset-description)
- [Dataset Summary](#dataset-summary)
- [Supported Tasks](#supported-tasks)
- [Languages](#languages)
- [Dataset Structure](#dataset-structure)
- [Dataset Creation](#dataset-creation)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Dataset Description

**Vi-ViDoRe** (Vietnamese Visual Document Retrieval) is the first governed, human-validated benchmark for visual document retrieval in Vietnamese. It evaluates multi-vector late-interaction models (ColPali, ColQwen) and text-based retrievers on real Vietnamese documents across legal, financial, healthcare, and education domains.

### Key Features

- **Governed data practices**: Document registry with provenance, license verification, split locking
- **Human-validated qrels**: Double annotation with Cohen's kappa ≥ 0.67, adjudication
- **Multi-domain**: Legal, Financial, Healthcare, Education
- **Visual + Text**: Native PDF pages as images + extracted text
- **Graded relevance**: 0 (not relevant), 1 (partially), 2 (fully relevant)
- **No contamination**: Source/template-held-out splits, deduplication audit

### Homepage

https://github.com/anonymous/vi-vidore

### Repository

https://github.com/anonymous/vi-vidore

### Paper

Under review. Preprint: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)

### Point of Contact

Anonymous (anonymous@example.com)

## Dataset Summary

| Split | Documents | Pages | Queries | Domains |
|-------|-----------|-------|---------|---------|
| Train | 5 | 50 | 57 | Education |
| Dev | 2 | 20 | 11 | Education |
| Test | 8 | 299 | 888 | Edu(30), Fin(51), Health(198), Legal(20) |

### Query Statistics (Test)

- **Total queries**: 888
- **Human-written**: ≥40% (target)
- **LMM-assisted**: Remaining
- **Domains**: Education (50+), Financial (50+), Healthcare (50+), Legal (50+)
- **Page types**: Text-heavy, Table-heavy, Chart-heavy, Form/Template, Scanned
- **Hardness**: Easy, Medium, Hard

### Document Sources

| Domain | Sources | License |
|--------|---------|---------|
| Education | Multiple universities | Various (institutional) |
| Legal | NASATI, Bộ Tư pháp | To be verified |
| Financial | World Bank, IMF, ADB | CC BY 3.0 IGO |
| Healthcare | World Bank, Bộ Y tế | CC BY 3.0 IGO |

## Supported Tasks

### Primary Task: Visual Document Retrieval

**Input**: Query text + Document images (multi-page PDFs)
**Output**: Ranked list of relevant page IDs
**Metrics**: nDCG@5, nDCG@10, MRR@10, Recall@5, Recall@10
**Evaluation**: Macro-domain average, per-domain, per-source-type, per-page-type

### Baselines Supported

| Model | Type | Zero-shot | Fine-tuned |
|-------|------|-----------|------------|
| BM25 (native text) | Lexical | ✓ | - |
| BGE-M3 / Vietnamese Bi-Encoder | Dense | ✓ | ✓ |
| ColPali (vidore/colpali-v1.2) | Multi-vector late interaction | ✓ | ✓ |
| ColQwen2 | Multi-vector late interaction | ✓ | ✓ |

### Evaluation Protocol

```python
from src.evaluation.evaluator import ViViDoReEvaluator
from src.evaluation.report_generator import save_evaluation_report

evaluator = ViViDoReEvaluator(split, pages_metadata, bootstrap_seed=42)
results = evaluator.evaluate_retrieval_results(retrieval_results, model_name="MyModel")
# results contains: overall CI, per-domain, per-source-type, per-query scores

# Significance testing
sig = evaluator.significance_test(results_a, results_b, metric="ndcg@5", method="paired_bootstrap")
# sig: p_value, mean_difference, significant_at_05, interpretation
```

## Languages

- **Primary**: Vietnamese (vi)
- **Query language**: Vietnamese
- **Document language**: Vietnamese
- **Evaluation**: Vietnamese queries on Vietnamese documents

## Dataset Structure

### Data Instances

#### Query
```json
{
  "query_id": "q_test_legal_001",
  "query_text": "Mức phạt vi phạm an toàn lao động theo Luật 2019",
  "domain": "legal",
  "query_type": "legal_clause",
  "source": "human_written",
  "target_page_ids": ["tl4_2021_p03", "tl4_2021_p04"],
  "hardness_level": "medium",
  "metadata": {
    "validation_status": "validated",
    "guideline_version": "1.0"
  }
}
```

#### Page Metadata
```json
{
  "doc_id": "tl4_2021",
  "page_num": 3,
  "page_id": "tl4_2021_p03",
  "file_path": "data/raw_pdfs/vn/tl4_2021.pdf",
  "image_path": "data/curated/pages/tl4_2021_p03.png",
  "sha256": "90b5160d8b544714b0e5ad0270d34e21a5dd1281826f996123e743acc938e75e",
  "phash": "a1b2c3d4...",
  "domain": "legal",
  "page_type": "text_heavy",
  "source_type": "born_digital",
  "native_text": "Điều 45. Mức phạt...",
  "char_count": 1245,
  "estimated_dpi": 150,
  "blur_score": 0.02
}
```

#### Qrel (Ground Truth)
```tsv
query_id    iteration    page_id    relevance
q_test_legal_001    0    tl4_2021_p03    2
q_test_legal_001    0    tl4_2021_p04    1
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | string | Unique identifier |
| `query_text` | string | Natural language query (stand-alone) |
| `domain` | enum | legal, financial, healthcare, education, infographic |
| `query_type` | enum | fact_lookup, legal_clause, numeric_table, multi_cell_comparison, chart_interpretation, paraphrase_or_abbreviation |
| `source` | enum | human_written, llm_assisted, heuristic_fallback |
| `target_page_ids` | list[string] | Relevant page IDs |
| `hardness_level` | enum | easy, medium, hard |
| `relevance` | int | 0 (not), 1 (partial), 2 (full) |

### Data Splits

| Split | Purpose | Query Source | Frozen |
|-------|---------|--------------|--------|
| Train | Model development, adapter training | Mixed | ✓ |
| Dev | Hyperparameter tuning, model selection | Mixed | ✓ |
| Test | Final evaluation, publication | Mixed | ✓ (after annotation) |

**Split integrity**: Source-held-out, template-held-out, no cross-split leakage.

## Dataset Creation

### Curation Pipeline

1. **Document Collection**: Source Vietnamese PDFs from verified publishers
2. **License Verification**: Confirm redistribution rights (CC BY, institutional)
3. **Processing**: PDF → images (150 DPI) + native text extraction (PyMuPDF)
3. **Deduplication**: SHA-256 exact + pHash near-duplicate detection
4. **Registry**: Authoritative CSV with provenance, license, split assignment
5. **Split Locking**: Source/template-held-out, frozen assignment hash
6. **Query Generation**: 
   - LLM-assisted (GPT-4o-mini / Gemini) with domain-specific prompts
   - Human-written by domain experts
   - Heuristic fallback (sanitized native text)
7. **Human Annotation**: 
   - Pool candidates: BM25 + Dense Vi + ColPali/ColQwen (RRF)
   - Double annotation, Cohen's kappa ≥ 0.67
   - Adjudication for disagreements
8. **Freeze**: Immutable manifest with all hashes (registry, criteria, metadata, queries, annotations)

### Annotation Guidelines

See `ANNOTATION_GUIDELINE_v1.0.md` for:
- Relevance scale (0/1/2) definitions
- Evidence types per domain
- Double annotation protocol
- Adjudication process

### Quality Assurance

- **Cohen's kappa**: ≥ 0.67 required
- **Agreement audit**: Per-domain, per-page-type breakdown
- **Contamination audit**: Exact duplicate, near-duplicate, source leakage
- **Domain balance**: ≥ 50 queries/domain, ≥ 20 queries/page-type
- **Human ratio**: ≥ 40% human-written queries

## Evaluation

### Official Metrics

| Metric | Description | Aggregation |
|--------|-------------|-------------|
| `macro_domain_ndcg@5` | Mean nDCG@5 across 4 domains | Primary |
| `overall_ndcg@5` | nDCG@5 with 95% CI (bootstrap) | Secondary |
| `mrr@10` | Mean Reciprocal Rank @10 | Secondary |
| `per_domain_ndcg@5` | nDCG@5 per domain | Diagnostic |
| `per_source_type_ndcg@5` | Born-digital vs Scanned | Diagnostic |
| `per_page_type_ndcg@5` | Text/Table/Chart/Form | Diagnostic |

### Significance Testing

- **Method**: Paired bootstrap (10,000 iterations) or Randomization test
- **Unit**: Per-query differences
- **Correction**: Holm-Bonferroni for multiple comparisons
- **Reporting**: p-value, mean difference, 95% CI, effect size

### Hardware Reporting

All results must include:
- GPU model (e.g., NVIDIA RTX 3090 24GB)
- CUDA version, PyTorch version
- Batch size, latency (P50, P95)
- Index size, VRAM usage

## Citation

```bibtex
@misc{vi-vidore-2026,
  title={Vi-ViDoRe: A Governed Vietnamese Visual Document Retrieval Benchmark},
  author={Anonymous},
  year={2026},
  note={Under review},
  url={https://github.com/anonymous/vi-vidore}
}
```

## License

- **Code**: MIT License
- **Annotations (qrels)**: CC BY 4.0
- **Documents**: Per-document licenses (see `data/governance/document_registry.csv`)
  - World Bank documents: CC BY 3.0 IGO
  - Institutional documents: Verify individually

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0-candidate | 2026-08-27 | Governed candidate, freeze gates defined |
| 1.0.0 | TBD | Frozen benchmark, human-validated qrels, baselines |

## Contact

For questions about the benchmark, data licensing, or evaluation:
- GitHub Issues: https://github.com/anonymous/vi-vidore/issues
- Email: anonymous@example.com