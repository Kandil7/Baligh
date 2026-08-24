"""Modal training script for Baligh-1.7B.

Usage:
    modal run src/modal/train.py --stage cpt
    modal run src/modal/train.py --stage sft --base-model /vol/training/cpt/final
    modal run src/modal/train.py --stage cpt --resume  # auto-resume from latest

Security notes:
- subprocess with LIST args (no shell interpolation) — CLI params can no
  longer inject shell commands.
- The repo clone is checked out at a pinned branch; for production runs pin
  a commit SHA instead of a moving branch name.
"""

import modal
from src.modal.app import VOL_PATH, app, vol

REPO_URL = "https://github.com/Kandil7/Baligh.git"
# Pin the ref used for training runs; replace with a commit SHA to freeze.
REPO_REF = "develop"


def _clone_repo(project_dir: str) -> None:
    import subprocess
    from pathlib import Path

    if Path(project_dir).exists():
        return
    print(f"Cloning project ({REPO_REF})...")
    # List args, no shell: nothing user-supplied is ever interpolated.
    subprocess.run(
        ["git", "clone", "--branch", REPO_REF, REPO_URL, project_dir],
        check=True,
    )


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=86400,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-token", required=False),
    ],
)
def train(
    stage: str = "cpt",
    base_model: str | None = None,
    resume: bool = False,
    config: str | None = None,
):
    """Run CPT or SFT training on Modal GPU."""
    import os
    import subprocess

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"
    os.environ["WANDB_DIR"] = f"{VOL_PATH}/wandb"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    project_dir = f"{VOL_PATH}/Baligh"
    _clone_repo(project_dir)
    os.chdir(project_dir)

    subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)

    output_dir = f"{VOL_PATH}/training/{stage}"

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

    data_dir = f"{VOL_PATH}/data/train_ready/{stage}"
    cmd = [
        "python",
        "-m",
        f"src.scripts.run_{stage}",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
    ]
    if stage not in ("cpt", "sft"):
        raise ValueError(f"Unknown stage: {stage}")
    if base_model:
        cmd += ["--base-model", base_model]
    if config:
        cmd += ["--config", config]
    if resume_path:
        cmd += ["--resume", resume_path]

    print(f"Running: {cmd}")
    result = subprocess.run(cmd, check=False)
    exit_code = result.returncode

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
    import subprocess

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    project_dir = f"{VOL_PATH}/Baligh"
    _clone_repo(project_dir)
    os.chdir(project_dir)

    subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)

    output_dir = f"{VOL_PATH}/data/train_ready"
    cmd = [
        "python",
        "-m",
        "src.scripts.prepare_data",
        "--stage",
        stage,
        "--clean",
        "--output-dir",
        output_dir,
    ]
    print(f"Running: {cmd}")
    subprocess.run(cmd, check=True)

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

    print("\n" + "=" * 60)
    print(f"Checkpoints for {stage.upper()}")
    print("=" * 60)
    for cp in checkpoints:
        step = cp.get("step", "N/A")
        loss = cp.get("loss", "N/A")
        timestamp = cp.get("global_time", "N/A")
        print(f"  Step {step:>6} | Loss: {loss} | Time: {timestamp}")
    print("=" * 60 + "\n")


# Entrypoints for `modal run`
@app.local_entrypoint()
def main(
    stage: str = "cpt",
    base_model: str | None = None,
    resume: bool = False,
    config: str | None = None,
):
    train.remote(stage=stage, base_model=base_model, resume=resume, config=config)
