# Review: training/checkpoint core + utils (v0.1.0)

### Context

Baligh repo (D:\AI\Projects\LLM\Baligh), Arabic LLM fine-tuning (Qwen3-1.7B QLoRA r=16/a=16, CPT+SFT, Colab T4/Vast.ai target). Read-only review of src/baligh/training/* and src/baligh/utils/* requested by main agent.

### Explanation

CRITICAL: (1) Both trainers pass `evaluation_strategy` to TrainingArguments/SFTConfig, removed in transformers>=4.46 (repo pins >=4.48) -> TypeError at init on any fresh install. (2) SFTTrainer called with `tokenizer=` (renamed processing_class in trl 0.12; repo pins trl>=0.11 unpinned) and formatting_func=PromptFormatter whose __call__ returns tokenized dicts, not strings; dataset_text_field AND formatting_func both set. (3) Defaults bf16 + flash_attention_2 + bnb bf16 compute dtype are all T4-incompatible (sm_75: no bf16, FA2 needs sm_80+) while README/hardware.yaml target Colab T4. MAJOR: CheckpointManager.save_checkpoint persists weights only (trainer.save_model) - no optimizer/scheduler/trainer_state -> signal-handler checkpoints resume as step-0 warm start, breaking mid-epoch auto-resume contract; dual retention (HF save_total_limit + manager cleanup) can delete the last FULL checkpoint keeping a weights-only newer one; run_sft --base-model accepted but silently ignored (CPT->SFT chain broken); prepare_model_for_kbit_training called twice; unsloth installed+documented but never imported; callbacks.py and metrics.compute_metrics defined but never registered (dead code, docs claim otherwise). MINOR: non-atomic metadata JSON crashes validate/list on corruption; signal handler reentrant on double Ctrl+C; best_metric logged as loss; PYTHONHASHSEED runtime no-op; determinism lacks CUBLAS_WORKSPACE_CONFIG; distributed.setup_distributed passes local_rank as global rank; logging.py import-time side effects; metrics NaN-unguarded, perplexity sequence-weighted not token-weighted. DONE WELL: seeding plumbed end-to-end (set_seed + seed/data_seed in args); checkpoint happy paths well unit-tested; hyperparams consistent across YAML/config/README (2e-4/1e-4 cosine, r16/a16); LoRA targets correct for Qwen2.5; coherent CheckpointManager API. VERDICT: REWORK - version-pin blockers + T4 dtype/attn defaults + broken stage handoff.

### Rationale (Why this?)

Findings verified against source line-by-line plus official sources: transformers PR #30190 deprecation (removal announced 4.46), 2025 forum/GitHub reports of TypeError, trl commit 47d08a9 renaming tokenizer->processing_class (v0.12), current TRL SFTTrainer signature (formatting_func: Callable[[dict], str]). T4 = Turing sm_75 facts from NVIDIA specs. Revisit if requirements are hard-pinned to older versions or notebooks override MIXED_PRECISION/ATTN env vars (not verified in notebook cells).

### Next Steps

Builder should: pin exact versions (transformers, trl) or migrate APIs; switch eval_strategy; fix SFTTrainer integration (processing_class, single formatting path); add GPU-capability gate for bf16/fa2 with fp16+sdpa T4 profile; make signal handler delegate to control.should_save for full-state checkpoints; thread base_model_path through train_sft; register or delete callbacks/compute_metrics.

---
