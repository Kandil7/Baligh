# Colab CLI for Baligh-1.5B v0

Run training on Google Colab free GPUs with Hugging Face Hub for storage (no Google Drive needed).

## Quick Start

```bash
# List available stages
python colab_cli/run.py list

# View notebook info
python colab_cli/run.py cpt

# Open in browser (opens Colab with notebook)
python colab_cli/run.py cpt --open

# Get upload instructions
python colab_cli/run.py cpt --upload
```

## Available Stages

| Stage | Notebook | Description |
|-------|----------|-------------|
| `cpt` | `colab_cpt_hf.ipynb` | Continued Pretraining only |
| `sft` | `colab_sft_hf.ipynb` | Supervised Fine-Tuning only |
| `eval` | `colab_eval_hf.ipynb` | Evaluation only |
| `full` | `colab_full_pipeline_hf.ipynb` | **Complete pipeline (recommended)** |

## How It Works

1. **Storage**: All artifacts saved to Hugging Face Hub (not Google Drive)
2. **Authentication**: HF_TOKEN entered via `getpass` in notebook
3. **Persistence**: Models, checkpoints, eval results pushed to HF repo
4. **Resume**: Clone from HF to continue interrupted runs

## Running on Colab

### Option 1: Direct Link (Easiest)
Click the direct Colab link:
```
https://colab.research.google.com/github/Kandil7/Baligh/blob/develop/colab_cli/colab_full_pipeline_hf.ipynb
```

### Option 2: Upload Manually
1. Go to https://colab.research.google.com/
2. File → Upload notebook → Select `.ipynb` file
3. Runtime → Change runtime type → **GPU (T4 or L4)**
4. Run all cells

### Option 3: CLI
```bash
python colab_cli/run.py full --open
```

## Required Secrets

Add to your HF repo settings → Secrets:
- `HF_TOKEN` - Write access token (for pushing models)

## Pipeline Flow (Full)

```
1. Install deps (Unsloth, Transformers, HF Hub)
2. Login to HF (enter token)
3. Clone project from HF/GitHub
4. Prepare data (CPT + SFT) -> data/train_ready/
5. CPT Training -> Push to HF /cpt
6. SFT Training -> Push to HF /sft  
7. Merge LoRA -> Push to HF /instruct
8. Quantize (GGUF q4_k_m) -> Push to HF /gguf
9. Evaluation -> Push to HF /eval
```

## Resume Interrupted Run

If Colab disconnects, just re-run the notebook - it will:
1. Clone latest from HF repo
2. Resume from last checkpoint (trainer auto-resumes)
3. Continue training

## Tips

- **T4 (16GB)**: Use batch_size=1, grad_accum=8 for CPT
- **L4/A100**: Use batch_size=2, grad_accum=4 (default)
- **Enable HF Transfer**: `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"` (faster uploads)
- **Monitor**: Check WandB/TensorBoard in notebook outputs
