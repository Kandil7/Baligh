# Decision: Base model pivot to unsloth/Qwen3-1.7B-Base (Baligh-1.7B)

**Date:** 2026-08-24
**Status:** Accepted

## Context

Baligh v0 was originally specced around Qwen2.5-1.5B. The project owner
redirected v0 to fine-tune `unsloth/Qwen3-1.7B-Base` instead, renaming the
effort Baligh-1.7B.

## Decision

1. `ModelConfig.model_name = "unsloth/Qwen3-1.7B-Base"` is the single
   source of truth for the base model. The previous dual field
   (`model_name` vs `unsloth_model_name`) is removed — two fields that can
   drift was a reviewed defect.
2. All defaults, configs, Modal image pins, workflows and docs reference
   Baligh-1.7B / Qwen3.
3. Dependency floors updated: `transformers>=4.51` (Qwen3 architecture
   support), `torch>=2.6` (CVE-2025-32434).
4. The chat-template fallback in `models/tokenizer.py` is now load-bearing:
   BASE Qwen3 tokenizers ship no chat template, so SFT formatting uses our
   valid-Jinja ChatML fallback (non-thinking format).

## Consequences

- Qwen3-1.7B specs: 28 layers, GQA 16Q/8KV, hidden 2048, 32K native context,
  36T-token pretraining corpus, Apache-2.0.
- LoRA target module names are identical to Qwen2.5 (q/k/v/o/gate/up/down),
  so the QLoRA recipe carries over unchanged.
- Effective context during v0 training remains 2,048 tokens; native 32K is
  an architecture property, not a trained one. Model card states this
  honestly; long-context training (RoPE scaling) is future work.
- fp16/sdpa auto-fallback on Turing remains required: Qwen3 does not change
  the sm_75 hardware constraint.

## Alternatives considered

- `Qwen/Qwen3-1.7B` (official repo): equally valid weights; rejected because
  the owner specified the unsloth mirror and notebooks already standardize
  on unsloth identifiers.
- Keep Qwen2.5-1.5B: rejected per owner direction.
