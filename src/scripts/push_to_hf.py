"""Push a model to the Hugging Face Hub for Baligh-1.7B."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Push model to HF Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--repo-id", type=str, required=True, help="HF repo ID (e.g., user/Baligh-1.7B-v0-instruct)"
    )
    parser.add_argument(
        "--token", type=str, default=None, help="HF token (defaults to cached login / HF_TOKEN)"
    )
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument(
        "--commit-message", type=str, default="Release Baligh-1.7B v0", help="Commit message"
    )
    args = parser.parse_args()

    setup_logging()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. Check before creating the remote repo."
        )

    api = HfApi(token=args.token)

    logger.info(f"Creating repo: {args.repo_id}")
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)

    logger.info(f"Uploading model from {args.model_path}")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    logger.info(f"Model pushed to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
