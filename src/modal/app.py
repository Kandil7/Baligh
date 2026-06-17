"""Modal deployment for Baligh-1.5B training and inference.

Usage:
    # Install Modal
    pip install modal
    modal setup  # authenticate

    # Run CPT training
    modal run src/modal/train.py --stage cpt

    # Run SFT training
    modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final

    # Run inference
    modal run src/modal/infer.py --prompt "ما هي عاصمة مصر؟"

    # Deploy as web endpoint
    modal deploy src/modal/serve.py
"""

import modal

# Custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.5.1",
        "transformers>=4.48.0",
        "accelerate>=0.37.0",
        "peft>=0.14.0",
        "trl>=0.11.0",
        "datasets>=2.21.0",
        "huggingface-hub>=0.26.0",
        "bitsandbytes>=0.44.0",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.5.0",
        "pyyaml>=6.0",
        "rich>=13.7.0",
        "loguru>=0.7.0",
        "wandb>=0.18.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
    )
    .pip_install(
        "unsloth @ git+https://github.com/unslothai/unsloth.git",
        gpu="a10g",
    )
)

# Persistent volume for training data and checkpoints
vol = modal.Volume.from_name("baligh-training", create_if_missing=True)
VOL_PATH = "/vol"

# Create app
app = modal.App("baligh", image=image)
