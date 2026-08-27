# Contributing to Vi-ViDoRe

Thank you for your interest in contributing! This document outlines the contribution workflow.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU experiments)
- Git
- Docker (optional, for containerized runs)

### Quick Start

```bash
# Clone and setup
git clone https://github.com/anonymous/vi-vidore.git
cd vi-vidore

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.lock
pip install -e .

# Run tests
python -m pytest tests/ -v
```

## Development Setup

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run on commit:
- `ruff` - linting
- `black` - formatting (line length 100)
- `mypy` - type checking
- `pytest` - unit tests (fast subset)

### Docker Development

```bash
# Build and run Jupyter
docker-compose up vi-vidore-jupyter

# Run baselines
docker-compose up vi-vidore-eval

# Build governed benchmark
docker-compose up vi-vidore-build
```

## Contribution Types

### 🐛 Bug Reports

Use the issue template with:
- Minimal reproduction case
- Environment info (`python -m pip list`, GPU, OS)
- Expected vs actual behavior

### 💡 Feature Requests

Describe the use case, proposed API, and any alternatives considered.

### 📝 Documentation

- Update docstrings for new functions
- Update `README.md` for new commands
- Add examples for new features

### 🧪 Benchmark Contributions

**Adding new documents:**
1. Add PDF to `data/raw_pdfs/vn/` or `data/raw_pdfs/cleared/`
2. Update `data/governance/document_registry.csv` with full metadata
3. Run `python scripts/05_build_governed_benchmark.py --reset-output`
4. Verify freeze gates pass

**Adding human queries:**
1. Use `scripts/balance_domains.py` template
2. Follow `ANNOTATION_GUIDELINE_v1.0.md`
3. Double-annotate with Cohen's kappa ≥ 0.67

**Adding baselines:**
1. Implement in `src/models/`
2. Register in `scripts/03_run_baselines.py`
3. Run significance tests via `evaluator.significance_test()`

### 🔬 Model Contributions

**Fine-tuning:**
1. Use `scripts/04_train_adaptation.py`
2. Log: seed, config, hardware, checkpoint revision
3. Report: ViDoRe retention, ≥3 seeds, OOD eval

## Pull Request Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feat/your-feature`
3. **Commit**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `refactor:` code restructuring
   - `test:` adding tests
   - `bench:` benchmark related
4. **Push** and open PR
5. **CI checks** must pass:
   - `ruff` lint
   - `black` format
   - `mypy` type check
   - `pytest` unit tests
   - `pytest tests/test_data_governance.py` (governance)
6. **Review** by maintainers
7. **Merge** after approval

## Coding Standards

### Python Style

- **Formatter**: `black` (line-length=100)
- **Linter**: `ruff` (pyproject.toml config)
- **Type checker**: `mypy` (strict mode for new code)
- **Docstrings**: Google style

```python
def function_name(param: Type) -> ReturnType:
    """Short description.

    Longer description if needed.

    Args:
        param: Description of param.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something goes wrong.
    """
```

### Type Hints

- All public functions must have type hints
- Use `from __future__ import annotations`
- Prefer `list[T]` over `List[T]` (Python 3.9+)

### Imports

```python
# Standard library
import json
from pathlib import Path

# Third-party
import torch
import numpy as np

# Local
from src.config import PathConfig
from src.data.schema import QueryItem
```

### Naming

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Test Categories

```bash
# Governance (critical for benchmark integrity)
python -m pytest tests/test_data_governance.py -v

# Metrics correctness
python -m pytest tests/test_metrics.py -v

# MaxSim implementation
python -m pytest tests/test_maxsim.py -v

# Schema validation
python -m pytest tests/test_schema.py -v

# Query sanitization
python -m pytest tests/test_query_sanitizer.py -v

# End-to-end pipeline
python -m pytest tests/test_end_to_end_pipeline.py -v
```

### Writing Tests

- Place in `tests/`
- Name: `test_<module>_<functionality>.py`
- Use fixtures from `tests/conftest.py`
- Mock external APIs (LLM, network)

## Documentation

### Update When

- Adding new CLI commands → `README.md`
- New model/retriever → `README.md` + docstring
- New evaluation metric → `src/evaluation/metrics.py` docstring
- New governance rule → `data/governance/FREEZE_CRITERIA.json` + `README.md`

### Diagram Updates

Architecture diagrams in `docs/` (if exists) should be updated for structural changes.

## Release Process

1. Update version in `src/config.py` / `pyproject.toml`
2. Update `CHANGELOG.md`
3. Tag: `git tag v0.x.y`
4. Build Docker: `docker build -t vi-vidore:v0.x.y .`
5. Push tag: `git push origin v0.x.y`

## Questions?

Open a GitHub Discussion or Issue. Maintainers will respond within 2 business days.

---

**Note**: This is a research benchmark. Data governance (licenses, annotations, splits) takes priority over code features. All benchmark-affecting changes require governance review.