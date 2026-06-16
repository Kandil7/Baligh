# Code Quality Standards - Baligh-1.5B v0

## Testing Strategy

### Test Pyramid

```
        E2E Tests (1)
           /    \
    Integration (5)
         /      
    Unit Tests (50+)
```

### Test Organization

```
tests/
├── unit/
│   ├── test_config.py           # Config validation, defaults
│   ├── test_cleaner.py          # Arabic normalization, quality filters
│   ├── test_mixer.py            # Ratio mixing, interleave behavior
│   ├── test_formatter.py        # CPT packing, SFT chat template
│   ├── test_validators.py       # Schema validation
│   ├── test_lora_config.py      # LoRA config creation
│   └── test_utils.py            # Seeding, memory, distributed
├── integration/
│   ├── test_data_pipeline.py    # End-to-end prepare_data
│   ├── test_cpt_training.py     # CPT trainer smoke test
│   ├── test_sft_training.py     # SFT trainer smoke test
│   └── test_evaluation.py       # Evaluator + benchmarks
└── fixtures/
    ├── sample_cpt_data.jsonl
    ├── sample_sft_data.jsonl
    └── sample_eval_data.jsonl
```

---

### Test Commands

```bash
# All tests with coverage
make test
# or
pytest tests/ -v --cov=src/baligh --cov-fail-under=80

# Unit only (fast)
pytest tests/unit -v

# Integration (requires GPU)
pytest tests/integration -v --gpu

# Specific module
pytest tests/unit/test_cleaner.py -v

# With coverage report
pytest tests/ --cov=src/baligh --cov-report=html:htmlcov
```

---

### Test Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `config.py` | 100% | Critical; all paths must work |
| `data/cleaner.py` | 95% | Core pipeline; Arabic normalization critical |
| `data/mixer.py` | 90% | Ratio mixing must be exact |
| `data/formatter.py` | 95% | Tokenization correctness critical |
| `data/validators.py` | 90% | Schema validation must pass |
| `training/*` | 80% | Trainers tested via integration |
| `evaluation/` | 85% | Metrics must be accurate |
| `models/loader.py` | 85% | Model loading critical path |

**Overall Target**: 80% (enforced by `--cov-fail-under=80`)

---

### Test Patterns

#### Unit Test Pattern (AAA)
```python
def test_normalize_arabic_alef_variants():
    # Arrange
    from baligh.data.cleaner import normalize_arabic
    
    # Act
    result = normalize_arabic("أ إ آ ا")
    
    # Assert
    assert result == "ا ا ا ا"
```

#### Integration Test Pattern
```python
@pytest.mark.integration
@pytest.mark.gpu
def test_cpt_trainer_smoke():
    # Uses max_samples=100 for fast test
    config = CPTConfig(max_steps=10, ...)
    trainer = CPTTrainer(config=config, train_dataset=tiny_dataset)
    result = trainer.train()
    assert result.training_loss < 10.0  # Sanity check
```

#### Fixtures
```python
# conftest.py
@pytest.fixture
def sample_cpt_data():
    return Dataset.from_dict({"text": ["نص تجريبي"] * 100})

@pytest.fixture
def sample_sft_data():
    return Dataset.from_dict({
        "instruction": ["سؤال"] * 10,
        "input": [""] * 10,
        "output": ["جواب."] * 10
    })
```

---

## Linting & Formatting

### Toolchain
| Tool | Version | Config |
|------|---------|--------|
| **Ruff** | 0.6.0 | `pyproject.toml` - `select = ["E","W","F","I","N","UP","B","C4","T20","SIM","ARG"]` |
| **Black** | 24.10.0 | `line-length=100`, `target-version=['py311','py312']` |
| **MyPy** | 1.11.0 | `strict=True`, `disallow_untyped_defs=True` |

### Commands
```bash
# Format
make format
# or
black src/ tests/
ruff check --fix src/

# Lint
make lint
# or
ruff check src/

# Type Check
make typecheck
# or
mypy src/
```

### Ruff Rules Explained
| Code | Rule | Why Enabled |
|------|------|-------------|
| E, W | pycodestyle | Basic style errors/warnings |
| F | pyflakes | Undefined names, unused imports |
| I | isort | Import sorting |
| N | pep8-naming | Naming conventions |
| UP | pyupgrade | Modern syntax upgrades |
| B | flake8-bugbear | Common bugs |
| C4 | flake8-comprehensions | Unnecessary comprehensions |
| T20 | flake8-print | No print() in production code |
| SIM | flake8-simplify | Simplifiable expressions |
| ARG | flake8-unused-arguments | Unused function arguments |

---

## Type Checking (MyPy)

### Configuration (`pyproject.toml`)
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
strict_optional = true
```

### Type Hints Standards
```python
# Good: Full type hints
def load_dataset_by_name(
    name: str, 
    split: str | None = None, 
    streaming: bool | None = None, 
    **kwargs: Any
) -> Dataset: ...

# Good: Frozen dataclass with slots
@dataclass(frozen=True, slots=True)
class CPTConfig:
    max_steps: int = 50000
    learning_rate: float = 2e-4

# Avoid: Untyped defs (MyPy error)
def bad_example(x):  # ERROR: missing type hints
    return x * 2
```

---

## CI/CD Pipeline (GitHub Actions)

### Workflows (`.github/workflows/`)

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | Push/PR to main/develop | Lint → Typecheck → Unit Tests → Coverage |
| `cpt-training.yml` | Manual dispatch | CPT training (self-hosted GPU) |
| `sft-training.yml` | Manual dispatch | SFT training (self-hosted GPU) |
| `release.yml` | Manual dispatch (version tag) | Merge → Quantize → Push HF → GitHub Release |

### CI Pipeline (`ci.yml`)
```yaml
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: pip install -r requirements/dev.txt && pip install -e .
      - run: ruff check src/
      - run: black --check src/
      - run: mypy src/
      - run: pytest tests/ -v --cov=src/baligh --cov-fail-under=80
```

### Pre-commit Hooks (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/psf/black
    hooks:
      - id: black
```

---

## Documentation Standards

### Docstring Style (NumPy/Google Hybrid)
```python
def load_base_model(
    model_name: str | None = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load base model with optional quantization.
    
    Args:
        model_name: Model name or path. Uses config default if None.
        load_in_4bit: Whether to load in 4-bit quantization.
        device_map: Device mapping strategy.
        
    Returns:
        Loaded model.
        
    Raises:
        ValueError: If model_name not found on HF Hub.
    """
```

### Module Docstrings
```python
"""Model loading utilities for Baligh-1.5B v0.

Handles 4-bit QLoRA loading, LoRA application/merging, 
tokenizer loading, and model preparation for training.
"""
```

---

## Security Practices

### Secrets Management
- **`.env` in `.gitignore`** - Never committed
- **GitHub Secrets** - `HF_TOKEN`, `WANDB_API_KEY` for CI/CD
- **`SettingsConfigDict(extra="ignore")`** - Ignores unknown env vars

### Dependency Security
```bash
# Scan for vulnerabilities
pip-audit
safety check

# In CI
- run: pip-audit --desc --format=json
```

### Container Security
```dockerfile
# Non-root user
RUN addgroup -g 1001 -S appgroup && adduser -S appuser -u 1001
USER appuser

# Read-only filesystem where possible
# No secrets in image (injected at runtime)
```

---

## Code Review Checklist

### For Every PR
- [ ] All tests pass (`make test`)
- [ ] Linting clean (`make lint`)
- [ ] Type check passes (`make typecheck`)
- [ ] Coverage ≥ 80% (enforced by CI)
- [ ] No `print()` statements (use `logger`)
- [ ] No hardcoded paths (use `constants.py`)
- [ ] No magic numbers (use `constants.py` or config)
- [ ] Docstrings for public functions/classes
- [ ] Type hints on all public functions
- [ ] No `any` type without justification
- [ ] No `type: ignore` without comment
- [ ] Config changes reflected in YAML configs
- [ ] Breaking changes documented in CHANGELOG

### For ML-Specific PRs
- [ ] Config changes reflected in YAML configs
- [ ] New datasets added to `DATASET_REGISTRY` with metadata
- [ ] New metrics added to `evaluation/metrics.py`
- [ ] Benchmark changes documented in `docs/evaluation/`
- [ ] Model card template updated if metrics change
