# Baligh-1.7B Runbook

Operational guide for running the Baligh training pipeline on the RTX 5000 (sm_75).
For architecture and design rationale see `docs/architecture/` and `docs/training/`.

## 1. Environment

```bash
# Activate venv
.\.venv\Scripts\Activate.ps1

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Restore deps (if needed)
uv sync --extra dev --extra eval
```

**Critical:** always use `--extra dev --extra eval` or `uv sync` will silently
uninstall pytest, ruff, sacrebleu, etc.

## 2. Data Preparation

```bash
# Both stages, cleaned, bounded (recommended for this machine)
python -m src.scripts.prepare_data --stage both --clean --max-samples 5000 --output-dir data/train_ready

# CPT only
python -m src.scripts.prepare_data --stage cpt --clean --max-samples 5000 --output-dir data/train_ready

# SFT only
python -m src.scripts.prepare_data --stage sft --clean --max-samples 5000 --output-dir data/train_ready
```

**Notes:**
- CPT uses streaming (ArabicWeb24, mC4, OSCAR, etc.) — requires network.
- `--max-samples` caps each source; omitting it materializes the full corpus
  (tens of GB, will OOM on 32 GB).
- SFT sources are standardized to `{instruction, input, output}` before mixing.
- Output: `data/train_ready/{cpt,sft}` with `train`/`eval` splits.

## 3. Training

### CPT (Continued Pretraining)

```bash
python -m src.scripts.run_cpt \
  --data-dir data/train_ready/cpt \
  --output-dir training/cpt \
  --config configs/cpt/cpt-stage1.yaml
```

- Auto-resumes from latest checkpoint if `training/cpt/` exists.
- Resume from specific checkpoint: add `--resume training/cpt/checkpoint-5000`.
- Output: `training/cpt/final/` (LoRA adapter + tokenizer).

### SFT (Supervised Fine-Tuning)

```bash
python -m src.scripts.run_sft \
  --data-dir data/train_ready/sft \
  --base-model training/cpt/final \
  --output-dir training/sft \
  --config configs/sft/sft-stage1.yaml
```

`--base-model` is load-bearing: without it, SFT starts from the raw base model.

### DPO (Preference Alignment)

```bash
python -m src.scripts.run_dpo \
  --data-dir data/smoke_test/preference.jsonl \
  --base-model training/sft/final \
  --output-dir training/dpo \
  --config configs/dpo/dpo-smoke.yaml
```

## 4. Evaluation

```bash
# Sunni-core + CIDAR (recommended)
python -m src.scripts.run_eval \
  --model-path training/sft/final \
  --benchmarks sunni cidar \
  --max-samples 200 \
  --output-dir docs/evaluation/post_sft

# Sunni only
python -m src.scripts.run_eval \
  --model-path training/sft/final \
  --benchmarks sunni \
  --max-samples 100

# CIDAR only
python -m src.scripts.run_eval \
  --model-path training/sft/final \
  --benchmarks cidar \
  --max-samples 200
```

**Note:** `--benchmarks mmlu` is currently blocked by an upstream MMLU-Arabic
CSV schema mismatch (not our code).

## Uploading to Hugging Face

```bash
 py -m src.scripts.push_to_hf --model-path release/baligh-1.7b-v0-instruct --repo-id Kandil7/Baligh-1.7B
```

## 5. Checkpoint Management

```bash
# Auto-resume
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --resume

# List checkpoints
python -c "
from baligh.training.checkpoint import CheckpointManager
m = CheckpointManager('training/cpt')
for cp in m.list_checkpoints():
    print(f\"Step {cp['step']:>6} | Loss: {cp.get('loss', 'N/A')}\")
"
```

- Full-state checkpoints: weights + optimizer + scheduler + RNG + trainer state.
- Ctrl+C triggers a reentrancy-safe save before exit.
- Last 3 manager-saved checkpoints are kept.

## 6. Troubleshooting

### CUDA out of memory
- Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps`.
- Reduce CPT `max_seq_length` (default 2048 → 1024 in formatter).
- Disable eval (`eval_strategy: "no"`) during training to avoid eval-batch OOM.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Network timeouts during data prep
- Retry; HF downloads resume. Set `HF_HUB_ENABLE_HF_TRANSFER=0` if `hf_transfer`
  is not installed.
- Use `--max-samples` to bound download size.

### YAML config not applied (wrong max_steps / learning_rate)
- Check for UTF-8 BOM: `head -c 3 file.yaml | xxd` should not show `EF BB BF`.
- Use decimal notation for floats (`0.0002` not `2e-4`) — PyYAML parses
  scientific notation as strings.
- Run scripts load configs with `encoding="utf-8-sig"`.

### `uv sync` uninstalled dev tools
- Always use `uv sync --extra dev --extra eval`.

### Fast tokenizer label padding error
- `DataCollatorForLanguageModeling` calls `tokenizer.pad()` which fails on a
  `labels` field with fast tokenizers. Use a custom collator that pads
  `input_ids`/`attention_mask` via `tokenizer.pad` and labels manually with -100.

### GradScaler BFloat16 error on Turing (sm_75)
- DPO with 4-bit models + fp16 GradScaler fails. Disable mixed precision
  (`fp16=False, bf16=False`) for DPO.

## 7. Cleanup

```bash
# Remove smoke test artifacts
python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/smoke_test','training/smoke_cpt','training/smoke_sft','training/smoke_dpo']]"

# Remove eval logs
python -c "import pathlib; [pathlib.Path(f).unlink(missing_ok=True) for f in ['data/cpt.log','data/cpt.err','data/sft.log','data/sft.err','data/eval.log','data/eval.err']]"
```

## 8. Quick Reference

| Step | Command | Output |
|------|---------|--------|
| Data prep | `prepare_data --stage both --clean --max-samples 5000` | `data/train_ready/` |
| CPT | `run_cpt --data-dir ... --config configs/cpt/cpt-stage1.yaml` | `training/cpt/final` |
| SFT | `run_sft --data-dir ... --base-model training/cpt/final` | `training/sft/final` |
| DPO | `run_dpo --data-dir ... --base-model training/sft/final` | `training/dpo/final` |
| Eval | `run_eval --model-path ... --benchmarks sunni cidar` | `docs/evaluation/` |
| Merge | `merge_lora --base-model ... --adapter-path ...` | `release/` |
| Quantize | `quantize --model-path ... --method gguf` | `release/*-gguf` |
| Model card | `generate_model_card --output ...` | `README.md` |
| Push | `push_to_hf --model-path ... --repo-id ...` | HF Hub |
