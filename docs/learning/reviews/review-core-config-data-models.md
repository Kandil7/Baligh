# Review: config/constants/data/models core modules

### Context

Read-only review of D:\AI\Projects\LLM\Baligh (Arabic LLM fine-tuning, Qwen3-1.7B QLoRA). Scope: src/baligh/config.py, constants.py, data/{cleaner,datasets,formatter,loader,mixer,validators}.py, models/{loader,quantization,tokenizer}.py. Verified regex bytes, cross-module grep usage, test suite, pyproject pins, and docs claims against Turing (sm_75) hardware reality.

### Explanation

CRITICAL: (1) models/tokenizer.py:61-75 fallback chat template is invalid Jinja (`{{% for %}}` instead of `{% for %}`) — breaks whenever tokenizer.chat_template is None (base-model SFT path). (2) data/loader.py DATASET_REGISTRY holds only 3 CPT datasets while datasets.py DATASETS has the full registry; load_sft_datasets/load_eval_datasets route through load_dataset_by_name → ValueError for every SFT/eval dataset; leftover metadata keys (text_column/license/tokens_estimate/domain) are forwarded as **config into HF load_dataset → TypeError risk. (3) PromptFormatter.__call__ expects batched dict but sft_trainer.py:141 passes it as TRL formatting_func (single example) → zip iterates characters → garbage training data or TRL error. (4) Defaults bf16 compute dtype + flash_attention_2 are incompatible with BOTH targets (RTX 5000 and Colab T4 are sm_75; bf16/FA2 need sm_80+); docs admit T4 needs fp16 but no fallback is implemented. MAJOR: response_only_loss=True declared but formatter copies full labels (flag never read); SFTConfig max_steps=10000 silently overrides num_train_epochs=1.0 (HF precedence); quantize_gguf calls nonexistent llama_cpp.convert module; quantize_gptq has no calibration data; merge_lora on 4-bit model fails; 'islamic_qa' in SFT mix absent from registry (silent renormalization); summarization dataset column contract (instruction_column='text') ignored by formatter; fix_repeated_punctuation deletes all repeats (sub r"" not r"\1"); normalize_whitespace \s+→' ' flattens paragraph structure (newline regex dead); chat-template tokenized path applies no truncation (OOM risk >2048). MINOR: constants.py imported nowhere (dead, triple-duplicates config defaults); ARABIC_DIACRITICS_RE dead; CONTROL_CHARS_RE contains literal control chars in source (verified bytes: preserves \\t\\n\\r, misses NUL/C1); dedup collapses <3-word docs (empty MinHashes identical); is_arabic misses presentation forms; duplicated tokenizer loaders (models/loader.py vs models/tokenizer.py); unused GGUF/AWQ/GPTQ dataclasses + QuantizationConfig parallel truth; loose pins (trl>=0.11 unbounded, unsloth git-main); heavily mocked tests pass the broken Jinja template.

### Alternatives

Considered treating registry split as intentional (docs acknowledge 'second registry') — rejected because functional consequence is crash on all SFT/eval loads. Considered rating whitespace-flattening as intentional simplification — rejected for CPT corpora where document structure carries signal. Considered approving-with-changes overall — rejected because 4 critical findings block any end-to-end run.

### Rationale (Why this?)

Findings verified by byte-level regex inspection, repo-wide grep (constants.py zero imports; response_only_loss zero consumers), and cross-reading trainer consumption of formatter. Revisit after builder patches: re-run review on loader.py/tokenizer.py/formatter.py diffs and add a real-Jinja render test plus a smoke load_dataset test.

### Exercises

1. Render get_chat_template() through jinja2.Template to prove the syntax error. 2. Call load_sft_datasets() against a stubbed load_dataset to reproduce the ValueError. 3. Simulate TRL formatting_func with a single-example dict through PromptFormatter.__call__. 4. Add unit test asserting fix_repeated_punctuation('wow!!!') == 'wow!'. 5. Grep for consumers of response_only_loss/packing_max_length/calibration_samples after refactor.

### Next Steps

Hand findings to builder: unify dataset registries, fix Jinja template, implement completion-only masking or consume response_only_loss, gate precision/attn defaults on torch.cuda.get_device_capability(), repair exporters, delete or wire constants.py.

---
