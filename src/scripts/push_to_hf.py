"""Push model to Hugging Face Hub for Baligh-1.5B v0."""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from baligh.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Push model to HF Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID (e.g., user/baligh-1.5b-v0-instruct)")
    parser.add_argument("--token", type=str, help="HF token")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--commit-message", type=str, default="Release Baligh-1.5B v0", help="Commit message")
    args = parser.parse_args()
    
    setup_logging()
    
    api = HfApi(token=args.token)
    
    logger.info("Creating repo: %s" % args.repo_id)
    create_repo(args.repo_id, token=args.token, private=args.private, exist_ok=True)
    
    logger.info("Uploading model from %s" % args.model_path)
    upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        commit_message=args.commit_message,
    )
    
    logger.info("Model pushed to: https://huggingface.co/%s" % args.repo_id)

if __name__ == "__main__":
    main()
