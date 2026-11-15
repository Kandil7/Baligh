# Baligh Project Rules

Arabic LLM fine-tuning for Islamic knowledge domain.

## Dataset Format

- Format: Alpaca-style (instruction, input, output) or ShareGPT (conversations)
- Arabic text: always include both normalized and display forms
- Train/test split: 90/10, stratified by topic
- Never put the same hadith in both train and test

## Fine-Tuning

- Use QLoRA (4-bit NF4 + LoRA adapters) for RTX 5000 16GB
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- LoRA rank: 16-64 (start at 16, scale up if underfitting)
- Batch size: 1-2 with gradient checkpointing
- Max sequence length: 1024-2048 (depends on dataset)

## Training Config

- Framework: transformers + PEFT + bitsandbytes
- Optimizer: paged_adamw_8bit
- LR: 2e-4 with cosine schedule
- Warmup: 3-5% of steps
- Epochs: 2-3 (watch for overfitting)
- Logging: every 10 steps to W&B or MLflow
- Checkpoint: save every 200 steps

## Evaluation

- Loss curve: train vs eval divergence > 0.2 = overfit
- Arabic coherence: manual review of 10-20 samples
- Factual accuracy: compare against source texts
- Perplexity: on held-out Islamic knowledge subset
- Use `evaluation` skill for metrics framework

## Arabic Text Preprocessing

- Normalize before tokenization: remove tashkeel, normalize alef/ta-marbuta/ya
- Never strip diacritics from Quranic text in training data
- Keep hadith chain (isnad) intact in training examples
- Use `arabic-nlp` skill for normalization pipeline

## Model Selection

- Base model: `unsloth/Qwen3-1.7B-Base` (decided in
  `docs/decisions/2026-08-24-base-model-qwen3-1.7b.md`, pinned by
  `configs/base/model.yaml`)
- Must support QLoRA (4-bit quantization)
- Check tokenizer handles Arabic + English + numbers
- Verify padding token exists (add if missing)
- Qwen3 base tokenizers ship no chat template: SFT relies on the valid-Jinja
  ChatML fallback in `models/tokenizer.py`

## Artifacts

- Checkpoints: `D:\AI\models\baligh\checkpoints\`
- Datasets: `D:\AI\datasets\baligh\`
- Logs: `D:\AI\experiments\baligh\`
- Never commit models or datasets to Git
