# Gotchas and Tips - Baligh-1.5B v0

## Critical Gotchas

### 1. **Padding Side Must Be Right for Causal LM**
```python
# ❌ WRONG - Breaks causal attention
tokenizer.padding_side = "left"

# ✅ CORRECT - Right padding for causal LM
tokenizer.padding_side = "right"
```
**Why**: Left padding shifts tokens; causal mask assumes future tokens are on the right

---

### 2. **Tokenizer Must Have pad_token**
```python
# Qwen has no pad_token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Standard workaround
```
**Forgetting this** → `ValueError: pad_token_id must be set` during training

---

### 3. **Response-only Loss Requires Accurate Masking**
```python
# ❌ WRONG - No masking; learns to predict user tokens
labels = input_ids.copy()

# ✅ CORRECT - Mask user tokens
labels = input_ids.copy()
assistant_start = find_assistant_start(input_ids)
labels[:assistant_start] = -100  # CrossEntropyLoss ignores -100
```
**Failure Mode**: Model learns to generate user prompts; fails at instruction following

---

### 4. **Streaming Datasets Cannot Use `num_proc > 1`**
```python
# ❌ WRONG - Streaming + multiprocessing = deadlock
dataset.map(fn, batched=True, num_proc=8)

# ✅ CORRECT
dataset.map(fn, batched=True, num_proc=1)  # Streaming
# OR
dataset.map(fn, batched=True, num_proc=8)  # Non-streaming (full load)
```

---

### 5. **`interleave_datasets` Stops at First Exhausted**
```python
# With stopping_strategy="first_exhausted" (default)
# Stops when SMALLEST dataset exhausted
# ArabicText-Large (743K) limits total steps vs ArabicWeb24 (28B tokens)
```
**Fix**: Ensure smallest dataset is large enough, or use `"all_exhausted"`

---

### 6. **LoRA Target Modules Must Match Model Architecture**
```python
# ❌ WRONG - Misses MLP projections
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ✅ CORRECT - Include MLP (SwiGLU) projections
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```
**Missing MLP** → LoRA only adapts attention; misses MLP knowledge

---

### 7. **`load_in_4bit` Requires `prepare_model_for_kbit_training`**
```python
# ❌ WRONG - Gradients won't flow properly
model = load_base_model(load_in_4bit=True)

# ✅ CORRECT
model = load_base_model(load_in_4bit=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
```
**Why**: Casts norms to fp32, enables gradient checkpointing for k-bit

---

### 8. **SFT `packing=False` is Mandatory**
```python
# ❌ WRONG - Packing mixes conversation turns
SFTConfig(packing=True, ...)

# ✅ CORRECT
SFTConfig(packing=False, ...)
```
**Why**: SFT conversations have semantic structure (user→assistant); packing concatenates unrelated turns

---

### 9. **SFT Loss: `response_only_loss=True` is Critical**
```python
# In SFTConfig
response_only_loss: bool = True  # Must be True!
```
**Without it**: Model learns to generate user prompts; fails instruction following

---

### 10. **`load_in_4bit=False` for Merging**
```python
# ❌ WRONG - Merging quantized model loses precision
model = load_base_model(load_in_4bit=True)
merged = merge_lora(model)

# ✅ CORRECT - Load full precision for merging
model = load_base_model(load_in_4bit=False)
merged = merge_lora(model)
```
**Why**: Merging requires full precision weights; 4-bit quantization loses precision irreversibly

---

## Performance Tips

### 1. **Use `HF_HUB_ENABLE_HF_TRANSFER=1` for Fast Uploads**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
# Multi-part parallel uploads; 5-10x faster for large models
```

### 2. **Use `uv` Instead of `pip`**
```bash
# 10-100x faster installs
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r requirements/training.txt
```

### 3. **Enable `torch.compile` (PyTorch 2.0+)**
```python
# In training script
model = torch.compile(model, mode="reduce-overhead")
# 10-20% speedup on Ampere+ GPUs
```

### 4. **Use `pad_to_multiple_of=8` for Tensor Cores**
```python
# In SFT data collator
DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
# Aligns to 8 for Tensor Core efficiency (8x8x8 MMA)
```

### 5. **Monitor GPU Memory with Callbacks**
```python
# In callbacks.py
class MemoryCallback:
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            log_memory_stats(prefix=f"Step {state.global_step}")
            clear_memory()
```

---

## Debugging Common Issues

### Issue: `CUDA Out of Memory`
**Solutions** (in order):
1. `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
2. `max_seq_length=1024` (reduce from 2048)
3. `gradient_checkpointing=True` (already default)
4. Disable `packing` for CPT (reduces peak memory)
5. Use `deepspeed` ZeRO-3 (if multi-GPU)

### Issue: `Triton Compilation Error`
```bash
# Unsloth requires Triton
pip install triton
# Or use --torch-backend=auto with uv
uv pip install unsloth --torch-backend=auto
```

### Issue: `CUDA Version Mismatch`
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall matching PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# or cu118 for older drivers
```

### Issue: `WandB Login Failed`
```bash
# Set API key
export WANDB_API_KEY=your_key
wandb login

# Or in script
wandb.login(key=os.getenv("WANDB_API_KEY"))
```

### Issue: `push_to_hub` Timeout
```bash
# Enable HF Transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# Retry with larger timeout
api.upload_folder(..., commit_message="...", timeout=3600)
```

---

## Pro Tips

### 1. **Test Pipeline with `max_samples=1000` First**
```bash
python -m src.scripts.prepare_data --stage cpt --clean --max-samples 1000
python -m src.scripts.run_cpt --data-dir data/train_ready/cpt --config configs/cpt/cpt-stage1.yaml
```
**Saves hours** catching config/data bugs early

### 2. **Use Config Files for Reproducibility**
```bash
# Instead of CLI args
python -m src.scripts.run_cpt --config configs/cpt/cpt-stage2.yaml
```
**Benefits**: Version-controlled config; exact reproducibility; easy grid search

### 3. **Monitor Memory Every 100 Steps**
```python
# In MemoryCallback
if state.global_step % 100 == 0:
    log_memory_stats(prefix=f"Step {state.global_step}")
    clear_memory()
```
**Catches OOM early** - shows allocation trend before crash

### 4. **Use `load_best_model_at_end=True` for SFT**
```yaml
# In SFT config
load_best_model_at_end: true
metric_for_best_model: "eval_loss"
greater_is_better: false
```
**Restores best checkpoint automatically** - no manual checkpoint selection

### 5. **Push to HF with `HF_TRANSFER`**
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
python -m src.scripts.push_to_hf --model-path release/baligh-1.5b-v0-instruct --repo-id Kandil7/Baligh-1.5B
```
**10x faster uploads** for multi-GB models

### 6. **Validate Data Before Training**
```bash
# Quick sanity check
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/train_ready/cpt/train')
print(f'Examples: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'Sample: {ds[0]}')
"
```
**Catches formatting bugs** before expensive training

### 7. **Use `make` Commands for Consistency**
```bash
make data-cpt      # Prepare CPT data
make train-cpt     # Run CPT training
make train-sft     # Run SFT training
make eval          # Run evaluation
make merge         # Merge LoRA
make quantize      # Quantize model
make push          # Push to HF
```

### 8. **Colab: Use T4 for Free, A100 for Speed**
| GPU | VRAM | CPT Time | SFT Time | Cost |
|-----|------|----------|----------|------|
| **T4 (Free)** | 16GB | ~4 hrs | ~2 hrs | Free |
| **L4 (Colab Pro)** | 24GB | ~2 hrs | ~1 hr | $10/mo |
| **A100 40GB** | 40GB | ~1 hr | ~30 min | $1-2/hr |

---

## Environment Setup Checklist

### Local Development
- [ ] Python 3.11+
- [ ] CUDA 12.1 + cuDNN 8.9+
- [ ] `make install-dev` (all deps)
- [ ] `.env` with `HF_TOKEN`, `WANDB_API_KEY`
- [ ] `pre-commit install`

### Colab
- [ ] Runtime → GPU (T4/L4)
- [ ] `HF_TOKEN` in Colab secrets
- [ ] `WANDB_API_KEY` in Colab secrets
- [ ] Mount Google Drive for persistence (optional)

### Production GPU (A100/H100)
- [ ] CUDA 12.1 + drivers 535+
- [ ] `make install-training`
- [ ] NVIDIA Container Toolkit (for Docker)
- [ ] HF_TOKEN in GitHub Secrets
- [ ] WANDB_API_KEY in GitHub Secrets

---

## Further Reading

| Topic | Resource |
|-------|----------|
| QLoRA Paper | https://arxiv.org/abs/2305.14314 |
| Unsloth Docs | https://unsloth.ai/docs/ |
| TRL SFTTrainer | https://huggingface.co/docs/trl/sft_trainer |
| Flash Attention 2 | https://github.com/Dao-AILab/flash-attention |
| Arabic NLP | https://github.com/arbml/arabic-nlp-resources |
| Islamic NLP | https://github.com/arbml/islamic-nlp |
