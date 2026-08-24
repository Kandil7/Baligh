# Baligh-1.7B full build-out: all criticals fixed, gates green

### Context

Full remediation of D:\AI\Projects\LLM\Baligh after Aug-2026 REWORK review, plus mid-session pivot from Qwen2.5-1.5B to unsloth/Qwen3-1.7B-Base (user directive). User worked concurrently: black-format sweep, 1.5B-to-1.7B rebrand sweep, and two commits (aad582c, 4480d29).

### Explanation

Completed P0-P3 remediation: (1) All 11 verified criticals fixed - Jinja chat template now valid {% %} ChatML without trim-tags eating newlines; dataset registry unified (loader delegates to datasets.DATASETS); SFT rewritten on plain HF Trainer with pre-formatted datasets and true completion-only masking (-100 prompt prefix via prompt-vs-full tokenization); base_model_path threaded through SFT (CPT handoff restored); eval_strategy migration verified against REAL TrainingArguments from installed transformers 4.57; capability-gated fp16/sdpa via new utils/hardware.py; merge_lora loads adapter on fp16 base with TypeError guard; colab run.py syntax fixed. (2) Arabic metrics: custom ROUGE over [\u0600-\u06ff...]+ tokens, sacrebleu tokenize=ar, normalized EM (casefold+diacritics+punctuation+digits). (3) Checkpoint full-state saves (optimizer/scheduler/trainer_state/rng) with atomic metadata, marker-based single-owner retention, reentrancy-safe SIGINT. (4) Security: unsloth PyPI-pinned >=2026.6,<2027, torch>=2.6 CVE fix, transformers>=4.51 Qwen3 floor, Modal endpoints POST+x-api-key gated, subprocess list-args, .dockerignore, non-root Dockerfiles, workflow permissions blocks + step-scoped secrets. (5) TRL dependency removed from library path. Gates ALL GREEN: 260 tests pass, coverage 81.11% (gate 80), ruff clean, ruff format clean, mypy Success on 44 files (strict with documented temporary no-untyped-def relaxation for 16 modules - shrink-don't-grow list in pyproject). Key lesson: transformers>=4.51 moved generate() to GenerationMixin (PreTrainedModel.generate gone); PowerShell Set-Content corrupts UTF-8 (use python or Edit tool for text surgery); never run ruff --unsafe-fixes unreviewed (l->o corruption incident).

### Rationale (Why this?)

Every gate verified empirically on the live tree (pytest/ruff/mypy/py_compile). Concurrent-edit collisions were detected and repaired (stray Qwen3-1.7B line in config.py, double @classmethod, l->o corruption from external formatter). Remaining risk: trainers never executed end-to-end against real models (CPU env); mocked wiring tests validate constructor/args only. Revisit if transformers major bump changes GenerationMixin layout.

### Next Steps

Commit remaining 15 dirty files (user decision). Roadmap: complete annotation sweep to drop the 16-module disallow_untyped_defs override; generate uv lockfiles; long-context training (RoPE scaling) since v0 trains at 2K window vs 32K native; smoke-run 10-step CPT on Colab T4 before any long run; consider deleting docs/learning historical docs drift later.

---

## Baligh-1.7B: notebooks fixed, strict mypy achieved with zero overrides (2026-08-24)

### Context

Follow-up completion round after the main Baligh-1.7B build-out session. Goal: eliminate remaining roadmap items — Colab notebook install correctness, Modal README phantom references, and full strict-mypy coverage.

### Explanation

Completed in this round: (1) Notebook audit revealed all 4 Colab notebooks are thin wrappers calling src.scripts.* via git clone — they inherit unsloth/Qwen3-1.7B-Base from library config automatically, but installed requirements/training.txt pulling flash-attn (fails/pointless on T4 sm_75), deepspeed, unused unsloth → patched to requirements/base.txt + GGUF llama.cpp tooling note added to merge/quantize cell; README Unsloth-in-notebooks claim corrected. (2) src/modal/README.md rewritten: phantom infer.py refs removed, POST+x-api-key auth documented with curl examples, baligh-api-key secret setup added, architecture diagram updated. (3) Full annotation sweep (~90 defs across 17 modules via 6 scripted passes using exact-match replacement): mypy relaxation override for baligh core DELETED — pyproject now runs disallow_untyped_defs=true globally with zero per-module escapes (only numpy stub-parse skip + scripts CLI glue remain). Notable API fact: transformers>=4.51 moved generate() to GenerationMixin — PreTrainedModel.generate no longer exists for typing (runtime OK via inheritance); route calls through cast to GenerationMixin. PowerShell -c multiline strings mangle quotes — always use temp script files for text surgery on Windows.

### Rationale (Why this?)

All gates re-verified after every scripted pass: 260 tests / 81.25% cov / ruff clean / ruff-format clean / mypy Success strict / 76-file py_compile clean. The strict-mode milestone from the previous log entry is now CLOSED.

### Next Steps

Commit dirty tree (user decision). Optional future: uv lockfiles, RoPE-scaling long-context stage, 10-step T4 smoke run before real training spend.

---
