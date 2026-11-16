# Session: Baligh Training Pipeline — Make It Runnable

**Date:** 2026-11-15
**Repo:** D:\AI\Projects\Baligh
**Goal:** Complete the IslamWeb fatwa SFT integration and make the full training/eval pipeline runnable end-to-end on the RTX 5000.

## Context

Continuation of the 2026-11-14 session that integrated the IslamWeb fatwa dataset.
This session focused on environment restore (uv sync), smoke testing, baseline eval,
and fixing every blocker that prevented `run_cpt` / `run_sft` / `run_eval` from
actually executing on the Turing GPU.

## What was done

- Restored `torch 2.11.0+cu128` and dev/eval extras after `uv sync` dropped them.
- Fixed CPT data cleaning (filter-then-map instead of shorter-list map) and
  iterable materialization before `save_to_disk`.
- Fixed `standardize_cpt_dataset` to drop original columns via `remove_columns`.
- Fixed CPT collator: fast tokenizer cannot pad `labels`, so a custom collator
  pads `input_ids`/`attention_mask` via `tokenizer.pad` and labels with -100.
  Forced `dataloader_num_workers=0` because the closure collator is not picklable.
- Fixed SFT/DPO trainers: `output_dir` Path coercion, `num_train_epochs=0` when
  `max_steps>0` (newer transformers rejects `None`), DPO mixed precision off
  (GradScaler/bf16 incompatibility on Turing), unsupported TRL kwargs removed.
- Fixed `run_eval`: wired `sunni` benchmark + `--sunni-path`, fail-fast validation
  before model load.
- Fixed CIDAR split (`test` → `train`), BLEU tokenizer (`ar` → `intl`).
- Stripped UTF-8 BOM from all YAML configs (was causing `CPTConfig` to silently
  use defaults — max_steps 50000 instead of 1000).
- Converted learning rates in YAML from scientific notation to decimals (PyYAML
  parsed `2e-4` as a string, breaking the optimizer).
- Reduced CPT max seq length to 1024 for VRAM; batch_size 1.
- Smoke test (CPT→SFT→DPO, 10 steps each) passed end to end.
- Baseline eval recorded: CIDAR ROUGE-L 0.040/BLEU 2.28; Sunni auto 0/7.
- CPT stage 1 completed: 1000 steps, loss 2.153, saved `training/cpt/final`.
- SFT stage 1 launched (running in background, ~11h).

## Key insights (learned)

- `remove_columns` in `datasets.map()` affects only the OUTPUT; the mapper still
  reads the full input. Both standardizer branches must pass it explicitly.
- A `.map()` cannot change row count — returning a shorter list causes
  "Column lengths mismatch". Use `.filter()` first, then `.map()`.
- Fast tokenizers cannot pad a `labels` field via `tokenizer.pad()` — it must be
  padded manually with the ignore index.
- PyYAML parses `2e-4` as a string, not a float. Always use decimal notation
  in YAML configs or coerce in code.
- UTF-8 BOM in YAML files silently breaks key lookups (`'\ufeffcpt' != 'cpt'`).
- `num_train_epochs=None` crashes newer transformers Trainer (`None > 0`).
- DPO with `ref_model=None` on a 4-bit model + fp16 GradScaler fails on Turing;
  disabling mixed precision is the safe fallback.
- `uv sync` (without `--extra`) uninstalls optional dependency groups.

## Still fuzzy

- Whether the SFT response-only loss masking correctly handles the ChatML template
  with Qwen3's tokenizer (needs manual sample review).
- Whether `max_seq_length=1024` for CPT is enough for the domain; may need
  2048 for longer Islamic texts.
- MMLU-Arabic eval is blocked by an upstream CSV schema mismatch — not our code,
  but needs a workaround or alternative benchmark.

## Next time

Launch SFT post-training eval and generate the model card once SFT completes.
Then expand the Sunni benchmark (50–100 items/category) with human-reviewed
religious citations before the final eval.

## Files touched

- src/baligh/training/cpt_trainer.py, sft_trainer.py, dpo_trainer.py
- src/baligh/data/formatter.py, datasets.py, loader.py
- src/scripts/prepare_data.py, run_eval.py, run_cpt.py, run_sft.py, run_dpo.py
- src/baligh/evaluation/benchmarks.py, metrics.py
- configs/cpt/cpt-stage1.yaml, configs/sft/sft-stage1.yaml (and all YAML BOM strip)
- docs/plan/qwen3_1.7b_baligh_starter_plan.md, README.md, docs/training/*