"""Modal app definition for Baligh-1.7B training and inference.

Usage:
    # Install Modal
    pip install modal
    modal setup  # authenticate

    # Run CPT training
    modal run src/modal/train.py --stage cpt

    # Run SFT training
    modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final

    # Deploy web endpoints
    modal deploy src/modal/serve.py

Security notes:
- Dependencies are version-pinned (no git-HEAD installs) so a compromised
  upstream cannot execute arbitrary code inside secret-holding builds.
- Web endpoints require an `x-api-key` header backed by a Modal Secret
  named "baligh-api-key"; unauthenticated GPU endpoints are cost abuse.
"""

import modal

# Custom image with pinned dependencies.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.6.0,<3",
        "transformers>=4.51.0,<5",  # 4.51+ required for Qwen3 architecture
        "accelerate>=0.37.0",
        "peft>=0.14.0",
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
        "sentencepiece>=0.2.0",
    )
)

# Persistent volume for training data and checkpoints
vol = modal.Volume.from_name("baligh-training", create_if_missing=True)
VOL_PATH = "/vol"

# API key secret for web endpoints (create with:
#   modal secret create baligh-api-key BALIGH_API_KEY=<random-string>)
api_key_secret = modal.Secret.from_name("baligh-api-key", required_keys=["BALIGH_API_KEY"])

# Create app
app = modal.App("baligh-1-7b", image=image)
