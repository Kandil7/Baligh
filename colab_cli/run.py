#!/usr/bin/env python3
"""
Colab CLI Runner for Baligh-1.7B
Uses Hugging Face Hub for storage (no Google Drive)
"""

import argparse
import json
import sys
from pathlib import Path

NOTEBOOKS = {
    "cpt": "colab_cpt_hf.ipynb",
    "sft": "colab_sft_hf.ipynb",
    "eval": "colab_eval_hf.ipynb",
    "full": "colab_full_pipeline_hf.ipynb",
}

GITHUB_REPO = "Kandil7/Baligh"
COLAB_BASE = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/develop/colab_cli"


def main():
    parser = argparse.ArgumentParser(description="Colab CLI for Baligh-1.7B")
    parser.add_argument(
        "stage", choices=list(NOTEBOOKS.keys()) + ["list"], help="Training stage to run"
    )
    parser.add_argument("--repo", default="Kandil7/Baligh-1.7B", help="HF repo ID")
    parser.add_argument("--open", action="store_true", help="Open notebook in browser")
    parser.add_argument("--upload", action="store_true", help="Print upload instructions")

    args = parser.parse_args()

    if args.stage == "list":
        print("Available stages:")
        for key, value in NOTEBOOKS.items():
            print(f"  {key:6} -> {value}")
        return

    nb_path = Path(__file__).parent / NOTEBOOKS[args.stage]

    if not nb_path.exists():
        print(f"[ERROR] Notebook not found: {nb_path}")
        sys.exit(1)

    print(f"[INFO] Notebook: {nb_path}")
    print(f"[INFO] Stage: {args.stage}")
    print(f"[INFO] Repo: {args.repo}")

    if args.open:
        import webbrowser

        colab_url = f"{COLAB_BASE}/{NOTEBOOKS[args.stage]}"
        print(f"[INFO] Opening: {colab_url}")
        webbrowser.open(colab_url)

    elif args.upload:
        print("\n[INFO] To upload to Colab:")
        print("1. Go to https://colab.research.google.com/")
        print(f"2. File -> Upload notebook -> Select: {nb_path}")
        print("3. Runtime -> Change runtime type -> GPU (T4/L4/A100)")
        print("4. Run all cells")
        print("\n[INFO] Or use the direct Colab link:")
        print(f"   {COLAB_BASE}/{NOTEBOOKS[args.stage]}")

    else:
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)
        print(f"\n[INFO] Notebook cells: {len(nb['cells'])}")
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code" and cell["source"]:
                first_line = cell["source"][0].strip()
                if first_line.startswith("# @title"):
                    print(f"  Cell {i + 1}: {first_line[8:].strip()}")
                elif first_line.startswith("#"):
                    print(f"  Cell {i + 1}: {first_line.strip()}")


if __name__ == "__main__":
    main()
