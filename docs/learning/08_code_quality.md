# 08 — Code Quality: Baligh-1.7B v0

## Code Style

### Consistency

- All modules use `from baligh.utils.logging import get_logger` for logging
- All config access uses factory functions (`get_config()`, `get_model_config()`, etc.)
- All dataclasses use `frozen=True, slots=True` for immutability
- All public APIs are re-exported via package `__init__.py` files

### Naming Conventions

- **Files**: `snake_case.py` (e.g., `cpt_trainer.py`, `lora_config.py`)
- **Classes**: `PascalCase` (e.g., `CPTTrainer`, `DatasetMixer`, `PromptFormatter`)
- **Functions**: `snake_case` (e.g., `load_base_model`, `apply_lora`, `mix_cpt_datasets`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BASE_MODEL_NAME`, `CPT_MIX_RATIOS`)
- **Private methods**: `_leading_underscore` (e.g., `_setup_model`, `_create_training_args`)

### Type Annotations

- All function signatures use type hints
- Union types use `X | Y` syntax (Python 3.10+)
- Optional types use `X | None` syntax
- Return types are annotated on all public functions

## Error Handling

### Data Validation Layer

- `validators.py` catches malformed examples before they reach training
- Invalid examples are counted and logged but not always removed (configurable)
- Schema validation (`check_dataset_schema`) raises `ValueError` for missing columns

### Training Layer

- `MemoryCallback` prevents OOM by clearing GPU cache periodically
- `CheckpointCallback` ensures regular saves to prevent data loss
- `CPTTrainer.train()` catches and logs training exceptions

### Quantization Layer

- `quantize_gguf()` captures subprocess stderr and raises `RuntimeError` on failure
- `quantize_awq()` and `quantize_gptq()` let library exceptions propagate

## Testing

### Test Structure (expected)

```
tests/
├── test_config.py          # Config creation and validation
├── test_data/
│   ├── test_loader.py      # Dataset loading
│   ├── test_cleaner.py     # Cleaning pipeline
│   ├── test_mixer.py       # Ratio mixing
│   ├── test_formatter.py   # CPT/SFT formatting
│   └── test_validators.py  # Validation logic
├── test_training/
│   ├── test_cpt_trainer.py # CPT training flow
│   └── test_sft_trainer.py # SFT training flow
├── test_models/
│   ├── test_loader.py      # Model loading
│   └── test_tokenizer.py   # Tokenizer utilities
└── test_evaluation/
    ├── test_metrics.py     # ROUGE, BLEU, EM
    └── test_benchmarks.py  # Benchmark runners
```

### Testing Approach

- Unit tests for each module's core functions
- Integration tests for the data pipeline (with max_samples=10)
- Mock-based tests for model loading (avoid GPU requirement)
- Validation tests with known-good and known-bad examples

## Linting and Formatting

### Expected Tools

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for automated checks

### Code Quality Metrics

- All functions have docstrings (Google style)
- No unused imports (ruff enforces)
- Type annotations on all public APIs
- No mutable default arguments (frozen dataclasses prevent this)

## Documentation

### Inline Documentation

- Module-level docstrings describe purpose
- Class docstrings describe responsibility
- Function docstrings describe parameters, return values, and behavior
- Complex logic has inline comments explaining reasoning

### External Documentation

- `docs/architecture/` - System architecture diagrams
- `docs/training/` - Training guides and hyperparameter explanations
- `docs/evaluation/` - Evaluation methodology and benchmark descriptions
- `docs/release/` - Release process and versioning
- `docs/learning/` - This learning documentation (00-09)

## Security Considerations

- No hardcoded secrets (HF tokens come from env vars or CLI args)
- `.env` file is gitignored
- No eval() or exec() usage
- Subprocess calls in quantize_gguf use hardcoded commands (no user input injection)
- Model loading uses `trust_remote_code=True` (required for Qwen2.5, but noted as a risk)

## Performance Considerations

- Streaming mode for large datasets avoids memory explosion
- Packing in CPT eliminates padding waste (2-5x throughput improvement)
- `dataloader_num_workers=4` for parallel data loading
- `dataloader_pin_memory=True` for faster CPU-to-GPU transfer
- `dataloader_drop_last=True` avoids small final batches
- `gradient_checkpointing=True` trades 30% compute for 70% activation memory savings
- Fused AdamW optimizer is 15-20% faster on CUDA
- `pad_to_multiple_of=8` in SFT DataCollator for GPU tensor core efficiency
