# ADR: DPO/ORPO Stack — Wrap TRL's DPOTrainer

**Date:** 2026-08-25
**Status:** Accepted
**Scope:** Preference alignment stage (post-SFT)

## Context

Baligh v0 had CPT and SFT trainers on a plain transformers + PEFT path but
zero preference-alignment code. The v1 roadmap requires DPO/ORPO to sharpen
Islamic-answer quality against rejected variants.

## Decision

1. **Adopt `trl>=0.9` as a core dependency** and wrap its `DPOTrainer`
   inside `baligh.training.dpo_trainer.DPOTrainer`, mirroring the existing
   CPT/SFT trainer shape (CheckpointManager, signal handling, memory stats).
2. **All trainer kwargs flow through TRL's `DPOConfig`** (a
   TrainingArguments subclass). Passing `beta`/`loss_type` as bare
   constructor kwargs was removed in trl>=0.9 and raises TypeError.
3. **Dataset contract is string columns `prompt/chosen/rejected`.**
   TRL tokenizes internally using `processing_class`; Baligh does NOT
   pre-tokenize. `instruction` is accepted as an input alias and renamed by
   `normalize_preference_schema()`.
4. **Reference model defaults to `None` with a PEFT policy**: TRL computes
   reference logits via adapter-disabled forward passes, avoiding a second
   weight copy — decisive for the 16 GB RTX 5000 budget.
5. **`loss_type` choices restricted to sigmoid/hinge/ipo** in
   `DPOConfig` (KTO is a separate trainer in TRL, not a DPO loss value).

## Alternatives Considered

| Option | Verdict |
|---|---|
| Hand-roll DPO loss | ~300 lines of reference-logprob numerics; high bug surface |
| ORPO-only (no ref model) | Simpler VRAM story, but less headroom-comparable with published DPO baselines; reachable anytime via `reference_free=true` |
| Keep SFT-only | Loses the alignment stage the SOTA goal depends on |

## Consequences

- New dependency `trl` (pulls `datasets`-compatible stack; verified 1.10.0
  imports cleanly alongside transformers>=4.51).
- `run_dpo.py` warns loudly when `--base-model` is omitted (same
  load-bearing contract as `run_sft.py`).
- Escape hatch if VRAM-tight: set `reference_free=true` (ORPO-style).

## Revisit When

- TRL major version bumps break `DPOConfig` kwargs again.
- Moving DPO to cloud GPUs where full-model (non-LoRA) DPO becomes viable.
