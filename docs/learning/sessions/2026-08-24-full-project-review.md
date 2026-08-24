# Baligh full-project review Aug 2026 — REWORK verdict, 11 verified criticals

### Context

Full-project review of D:\AI\Projects\LLM\Baligh (Arabic LLM fine-tuning, Qwen3-1.7B, QLoRA) on branch develop @ 4a34ba7. Four parallel specialist reviews (data/models, training/utils, eval/inference/scripts, security/infra) plus independent verification via py_compile, ruff, and direct source reads.

### Explanation

Verdict: REWORK. Pipeline cannot run end-to-end: 11 verified blockers including colab_cli/run.py:55 SyntaxError (py_compile-confirmed), invalid Jinja fallback template {{% %}} (tokenizer.py:61-75), dataset registry split-brain (loader.py 3 CPT entries vs datasets.py 14 → all SFT/eval loading raises ValueError), sft_trainer.py:176 base_model_path accepted-but-ignored (CPT→SFT handoff silently disconnected), evaluation_strategy removed in transformers≥4.46 while floor is ≥4.48, TRL tokenizer=/formatting_func API breakage, bf16+flash_attention_2 defaults on sm_75 targets (T4/RTX5000), merge_lora never loads adapter, GGUF/GPTQ exporters broken. Metrics invalid for Arabic: rouge_score default regex [a-z0-9]+ strips all Arabic chars; EM unnormalized; sacrebleu missing tokenize="ar". Ruff: 214 errors incl. 38 PEP-701 f-strings breaking declared Python 3.11 floor → CI red. Security: history/tree free of secrets, but unsloth@git-main unpinned into secret-holding envs, unauthenticated public Modal endpoints, job-level secrets on self-hosted runners, no .dockerignore with COPY . . Strengths: frozen dataclass configs, end-to-end seeding, hyperparameter consistency, correct LoRA targets, clean notebook outputs. Full details in docs/learning/reviews/review-{security-infra,core-config-data-models,training-checkpoint-core,evaluation-inference-scripts}.md

### Rationale (Why this?)

Every verdict-critical claim was empirically verified (compilation, lint, direct reads); remaining findings carry agent-supplied file:line references labeled FACT vs INFERENCE. Repo shows no evidence a real training run ever completed — all blockers are startup/handoff failures. Revisit if transformers/trl get hard pins and a smoke train of 10 steps succeeds.

### Next Steps

P0 fixes (pins + API migration, registry unification, base_model_path threading, fp16/sdpa gate, run.py syntax) → P1 (Arabic metric tokenization, greedy seeded eval) → P2 (pin unsloth SHA, Modal auth, .dockerignore, permissions blocks) → P3 (214 lint errors, dead code removal, docs sync). Re-review after P0/P1 before any training spend.

---
