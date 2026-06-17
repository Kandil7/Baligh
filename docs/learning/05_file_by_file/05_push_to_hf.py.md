# push_to_hf.py - Push to Hugging Face Hub

**Path**: `src/scripts/push_to_hf.py` (37 lines)

## Purpose

CLI entry point for pushing models and artifacts to Hugging Face Hub.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --model-path | str | Yes | - | Path to model |
| --repo-id | str | Yes | - | HF repo ID (e.g., user/baligh-1.5b-v0-instruct) |
| --token | str | No | None | HF token (or set HF_TOKEN env var) |
| --private | flag | No | False | Make repo private |
| --commit-message | str | No | "Release Baligh-1.5B v0" | Commit message |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Create HF API instance
4. Create repo (if it doesn't exist)
5. Upload model folder to repo
6. Log success with repo URL

---

## Usage Examples

```bash
# Push merged model
python -m src.scripts.push_to_hf \
    --model-path release/baligh-1.5b-v0-instruct \
    --repo-id Kandil7/Baligh-1.5B-v0-instruct

# Push with private repo
python -m src.scripts.push_to_hf \
    --model-path release/baligh-1.5b-v0-instruct \
    --repo-id Kandil7/Baligh-1.5B-v0-instruct \
    --private

# Push GGUF quantized model
python -m src.scripts.push_to_hf \
    --model-path release/baligh-1.5b-v0-instruct-gguf \
    --repo-id Kandil7/Baligh-1.5B-v0-instruct-GGUF
```

---

## Environment Variables

- `HF_TOKEN`: Hugging Face authentication token (alternative to --token flag)

---

## Tips

```bash
# Enable fast uploads
export HF_HUB_ENABLE_HF_TRANSFER=1
```
