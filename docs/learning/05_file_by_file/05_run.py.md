# run.py - Colab CLI Runner

**Path**: `src/scripts/run.py` (colab_cli/)

## Purpose

Google Colab notebook runner for Baligh-1.7B v0 training and inference.

---

## Features

- Interactive CLI for Colab environments
- GPU detection and configuration
- Data preparation, training, evaluation, and inference
- Checkpoint management
- Hugging Face Hub integration

---

## Usage in Colab

```python
# Install dependencies
!pip install -r requirements/training.txt

# Run the CLI
!python colab_cli/run.py
```

---

## Colab Notebook Structure

```
colab_cli/
├── run.py              # Main CLI runner
├── 01_setup.ipynb      # Environment setup
├── 02_data_prep.ipynb  # Data preparation
├── 03_cpt.ipynb        # CPT training
├── 04_sft.ipynb        # SFT training
├── 05_eval.ipynb       # Evaluation
├── 06_inference.ipynb  # Inference demo
└── README.md           # Colab-specific documentation
```

---

## GPU Requirements

| GPU | VRAM | CPT Time | SFT Time | Cost |
|-----|------|----------|----------|------|
| T4 (Free) | 16GB | ~4 hrs | ~2 hrs | Free |
| L4 (Colab Pro) | 24GB | ~2 hrs | ~1 hr | $10/mo |
| A100 40GB | 40GB | ~1 hr | ~30 min | $1-2/hr |

---

## Tips for Colab

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **Use HF_TOKEN**: Store in Colab secrets for model uploads
3. **Mount Drive**: For persistent storage across sessions
4. **Use max_samples**: Test with small datasets first
5. **Save checkpoints**: To Google Drive for persistence
