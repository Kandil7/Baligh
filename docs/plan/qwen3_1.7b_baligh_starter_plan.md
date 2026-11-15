# Baligh-1.7B Starter Plan (Qwen3-1.7B)

**Date:** 2026-11-14
**Status:** Execution in progress — data stage complete, training not started
**Supersedes:** the Qwen2.5-1.5B framing in `docs/plan/عاوز اعمل Baligh-1.5B v0.md`

## Goal

Produce a first releasable checkpoint of **Baligh-1.7B v0**: an Arabic-first,
Islamic-knowledge-specialized assistant built on `unsloth/Qwen3-1.7B-Base`
via QLoRA, trained on this workstation's RTX 5000 (16 GB, sm_75).

Success for v0 means all five, in order:

1. The pipeline runs end to end (data -> CPT -> SFT -> DPO -> eval) with
   checkpoints that save and resume.
2. Eval beats the untouched base model on the `sunni` benchmark set.
3. No train/eval leakage in any Islamic source (checked mechanically).
4. The model card states capabilities and limits honestly.
5. Artifacts release under Apache-2.0 with a LICENSE that exists.

## Fixed decisions

| Decision | Source |
| --- | --- |
| Base model `unsloth/Qwen3-1.7B-Base`, single `model_name` field | `docs/decisions/2026-08-24-base-model-qwen3-1.7b.md` |
| QLoRA NF4, r=16, alpha=16, dropout 0.0, all 7 attention/MLP projections | `src/baligh/config.py:102-107` |
| Effective context 2,048 tokens (native 32K is untrained) | ADR consequences |
| fp16, never bf16 — sm_75 lacks bf16 tensor cores | `src/baligh/utils/hardware.py:46-58` |
| DPO via TRL | `docs/decisions/2026-08-25-dpo-trl-stack.md` |
| Apache-2.0 for code | `pyproject.toml:7`, `LICENSE` |

## Constraints

- **Hardware.** i7-9850H, 32 GB RAM, RTX 5000 16 GB (Turing, sm_75). No bf16,
  no flash-attention-2 on this GPU; `attn_implementation: null` auto-resolves
  to sdpa.
- **Storage.** Models, datasets and checkpoints live under `D:\AI\`, never in
  Git.
- **Bandwidth and power.** Egypt connection, AC power for GPU runs: prefer
  local unit tests and dry runs, push only verified steps to training.

## Stage plan

### Stage 0 — Environment (BLOCKED)

The venv currently holds CPU-only torch (`2.13.0+cpu`, `torch.cuda` False)
while `uv.lock` pins the cu128 wheel. The wheel is reachable (2.75 GB), so
`uv sync` is expected to restore it.

- [ ] `uv sync` installs cu128 torch
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` -> `True`
- [ ] `nvidia-smi` sees the device from inside the venv

### Stage 1 — Data (DONE, verified 2026-11-14)

- [x] Local Islamweb corpus converted: 46,501 raw -> 3,917 rows
      (946 ar + 2,971 capped multilingual), grouped split by `fatwa_id`,
      `group_overlap: 0`, `row_level_split_safe: true`, leakage **PASS**
- [x] Registry supports local JSON/JSONL sources with split mappings
      (`src/baligh/data/loader.py`)
- [x] All four SFT sources load and standardize to exactly
      `{instruction, input, output}`: cidar 10,000, evol 59,022,
      islamweb_fatwa 3,525, summarization 4,000
- [x] Mix verified: 23,239 rows, `strict=True`, stage-1 budget binds on
      islamweb_fatwa at 23,500 predicted
- [x] SFT cleaning no longer applies the Arabic-only gate (it blanked 2,668
      of 3,525 fatwa rows); CPT keeps the gate
- [x] Dry run `--clean --max-samples 200` exits 0 with all sources valid

Rebuild command:

```powershell
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m src.scripts.convert_islamweb
.venv\Scripts\python.exe -m src.scripts.prepare_data --stage sft --clean --output-dir data/train_ready
```

### Stage 2 — Smoke test (DONE 2026-11-15)

Playbook `prompt/07-smoke-test.md` executed end to end: 10-step CPT, SFT (with
base-model handoff) and DPO all completed; `run_eval` ran; checkpoint save/resume
verified; smoke artifacts cleaned up. Fixes applied during the run: CPT fast-
tokenizer label padding via a custom collator, SFT/DPO precision handling, and
`run_eval --benchmarks sunni` wiring.

### Stage 3 — Baseline evaluation (DONE 2026-11-15)

Base model `unsloth/Qwen3-1.7B-Base` (20 CIDAR samples, 16 Sunni items):
CIDAR ROUGE-L 0.040 / BLEU 2.28; Sunni auto 0/7 (quran 0/2, hadith 0/1,
rag 0/2, refusal 0/1, safety 0/1); 9 items human-queued. Report:
`docs/evaluation/baseline/eval_report_*.json`.

### Stage 4 — CPT training (DONE 2026-11-15)

1000 steps, final loss 2.153, saved `training/cpt/final`. Config
`configs/cpt/cpt-stage1.yaml` (batch_size 1, max_seq 1024, eval disabled for VRAM).

### Stage 5 — SFT training (IN PROGRESS 2026-11-15)

Running against `training/cpt/final`, config `configs/sft/sft-stage1.yaml`
(batch_size 1, grad_accum 8). Output: `training/sft/final`.

`run_eval --benchmarks sunni` is now reachable: the `sunni` choice and
`--sunni-path` flag were added, and both benchmark validations run before the
model loads. The benchmark file `data/benchmarks/sunni_core_v0.jsonl` exists.

- [ ] `run_eval --benchmarks sunni` on the untouched base model
- [ ] Record baseline numbers for `sunni_core_v0` in
      `docs/learning/sessions/` so post-training gains are comparable
- [ ] Expand the Sunni benchmark to 50-100 items per category

Without this baseline, "did training help" is unanswerable.

### Stage 4 — Training

| Run | Config | Declared mix | Steps | LR |
| --- | --- | --- | --- | --- |
| CPT stage 1 | `configs/cpt/cpt-stage1.yaml` | web corpora | 1,000 | 2e-4 |
| CPT stage 2 | `configs/cpt/cpt-stage2.yaml` | web corpora | 50,000 | 2e-4 |
| CPT Islamic cycle | `configs/cpt/cpt-islamic.yaml` | hadith/quran | 5,000 | 1e-4 |
| SFT stage 1 | `configs/sft/sft-stage1.yaml` | 40/35/15/10 | 5,000 | 1e-4 |
| SFT stage 2 | `configs/sft/sft-stage2.yaml` | 35/30/20/15 | 10,000 | 5e-5 |
| DPO stage 1 | `configs/dpo/dpo-stage1.yaml` | preference | 1,000 | 5e-5 |

Note: `get_sft_config()` returns a frozen default; editing only the YAML does
not change the trained mixture. Verify the loaded ratios at run start.

- [ ] SFT stage 1 completes and saves `training/sft_stage1/final`
- [ ] Loss curve reviewed for divergence (train vs eval > 0.2 = overfit)

### Stage 5 — Evaluation and review

- [ ] `run_eval --benchmarks sunni` on the trained model, delta vs Stage 3
- [ ] MMLU-Arabic, CIDAR ROUGE-L, Arabic perplexity
- [ ] Manual review of 10-20 Arabic samples plus fatwa-style answers
      (attribution wording, not flat assertions)
- [ ] `run_eval` numbers feed `generate_model_card.py`

### Stage 6 — Release

- [x] `LICENSE` exists (Apache-2.0, verified against apache.org text)
- [ ] README badge/links resolve
- [ ] Model card generated from live config (`src/scripts/generate_model_card.py`
      now derives the dataset mix from `get_sft_config()`)
- [ ] Push to `Kandil7/Baligh-1.7B` — owner decision, never automatic

## Current verified status

| Check | Result |
| --- | --- |
| `pytest tests` | 292 passed, coverage 82.5% (gate 80%), exit 0 |
| `ruff check src/` | clean |
| `black --check src/` | clean (45 files) |
| `mypy src/` | clean (49 files) |
| SFT dry run `--clean` | exit 0, `Saved sft: {'train': 496, 'eval': 11}` |
| Fatwa leakage | `group_overlap: 0`, `repeated_fatwa_ids: 0` |
| GPU training | **not run** — venv torch is CPU-only |
| LICENSE | present, matches canonical Apache-2.0 |

## Risks

- **Single-source Islamic data.** Both fatwa sources are islamweb.net. At
  15-20% mix weight the model's Islamic voice is one publisher's editorial
  line. Mitigation: attribution wording in the instruction, human-eval
  stratification, later multi-source fatwa intake.
- **Small binding dataset.** islamweb_fatwa (3,525 rows) binds the whole
  stage-1 mix at 23,500 rows. Raising its weight shrinks the total; lowering
  it wastes fatwa rows. Recheck the budget whenever ratios change.
- **Untrained 32K context.** Long fatwa answers are truncated at 8,000 chars
  in the converter and 2,048 tokens in training. Do not market long-context
  ability.
- **Sensitive content.** Fatwas cover intimate personal questions. Eval and
  release review must include these categories, not just generic Arabic QA.
- **License of fatwa text.** No explicit license on the source pages. Internal
  training is fine; redistribution of the dataset is not, and the model card
  must not imply the fatwa text is CC-licensed.

## Ordered next actions

1. `uv sync` to restore cu128 torch (needs approval — 2.75 GB download).
2. Run playbook 07 smoke test end to end.
3. Baseline `run_eval --benchmarks sunni`.
4. SFT stage 1 training run.
5. Post-training eval and delta report.
6. Model card + release checklist.
