"""Modal training script for Baligh-1.5B.

Usage:
    modal run src/modal/train.py --stage cpt
    modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final
    modal run src/modal/train.py --stage cpt --resume  # auto-resume from latest checkpoint
"""

import modal
from src.modal.app import app, vol, VOL_PATH


@app.function(
    gpu="a10g",  # or "a100" for larger GPUs
    volumes={VOL_PATH: vol},
    timeout=86400,  # 24 hours max
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-token", required=False),
    ],
    cloud="aws",
)
def train(
    stage: str = "cpt",
    base_model: str = None,
    resume: bool = False,
    config: str = None,
    max_steps: int = None,
):
    """Run CPT or SFT training on Modal GPU."""
    import os
    from pathlib import Path

    # Setup environment
    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"
    os.environ["WANDB_DIR"] = f"{VOL_PATH}/wandb"

    # Get HF token from secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Clone project if not mounted
    project_dir = Path(VOL_PATH) / "Baligh"
    if not project_dir.exists():
        print("Cloning project...")
        os.system(f"git clone https://github.com/Kandil7/Baligh.git {project_dir}")

    os.chdir(project_dir)

    # Install dependencies
    os.system("pip install -e . -q")

    # Determine output directory
    output_dir = f"{VOL_PATH}/training/{stage}"

    # Determine checkpoint resume path
    resume_path = None
    if resume:
        from baligh.training.checkpoint import CheckpointManager
        manager = CheckpointManager(output_dir)
        latest = manager.get_latest_checkpoint()
        if latest:
            resume_path = str(latest)
            print(f"Auto-resuming from: {resume_path}")
        else:
            print("No checkpoint found, starting from scratch")

    # Build command
    if stage == "cpt":
        data_dir = f"{VOL_PATH}/data/train_ready/cpt"
        cmd = f"python -m src.scripts.run_cpt --data-dir {data_dir} --output-dir {output_dir}"
        if config:
            cmd += f" --config {config}"
        elif max_steps:
            # Create temp config with custom max_steps
            cmd += f" --config configs/cpt/cpt-stage1.yaml"
    elif stage == "sft":
        data_dir = f"{VOL_PATH}/data/train_ready/sft"
        cmd = f"python -m src.scripts.run_sft --data-dir {data_dir} --output-dir {output_dir}"
        if base_model:
            cmd += f" --base-model {base_model}"
        if config:
            cmd += f" --config {config}"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if resume_path:
        cmd += f" --resume {resume_path}"

    print(f"Running: {cmd}")
    exit_code = os.system(cmd)

    # Commit volume to persist checkpoints
    vol.commit()

    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}")

    print(f"Training complete! Checkpoints saved to {output_dir}")
    return f"{output_dir}/final"


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def prepare_data(stage: str = "cpt"):
    """Prepare training data on Modal."""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    project_dir = Path(VOL_PATH) / "Baligh"
    if not project_dir.exists():
        os.system(f"git clone https://github.com/Kandil7/Baligh.git {project_dir}")

    os.chdir(project_dir)
    os.system("pip install -e . -q")

    output_dir = f"{VOL_PATH}/data/train_ready"
    cmd = f"python -m src.scripts.prepare_data --stage {stage} --clean --output-dir {output_dir}"
    print(f"Running: {cmd}")
    os.system(cmd)

    vol.commit()
    print(f"Data preparation complete! Output: {output_dir}")


@app.function(
    volumes={VOL_PATH: vol},
    timeout=300,
)
def list_checkpoints(stage: str = "cpt"):
    """List all checkpoints for a training stage."""
    from baligh.training.checkpoint import CheckpointManager

    output_dir = f"{VOL_PATH}/training/{stage}"
    manager = CheckpointManager(output_dir)
    checkpoints = manager.list_checkpoints()

    if not checkpoints:
        print(f"No checkpoints found for {stage}")
        return

    print(f"\n{'='*60}")
    print(f"Checkpoints for {stage.upper()}")
    print(f"{'='*60}")
    for cp in checkpoints:
        step = cp.get('step', 'N/A')
        loss = cp.get('loss', 'N/A')
        time = cp.get('global_time', 'N/A')
        print(f"  Step {step:>6} | Loss: {loss} | Time: {time}")
    print(f"{'='*60}\n")


# Entrypoints for `modal run`
@app.local_entrypoint()
def main(
    stage: str = "cpt",
    base_model: str = None,
    resume: bool = False,
    config: str = None,
):
    """Main entrypoint: modal run src/modal/train.py --stage cpt"""
    train.remote(stage=stage, base_model=base_model, resume=resume, config=config)
