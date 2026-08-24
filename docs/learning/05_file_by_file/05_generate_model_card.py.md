# generate_model_card.py - Model Card Generator

**Path**: `src/scripts/generate_model_card.py` (105 lines)

## Purpose

CLI entry point for generating Hugging Face model cards with evaluation scores.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --output | str | Yes | - | Output path for model card |
| --mmlu-score | str | No | "TBD" | MMLU-Arabic score |
| --cidar-rouge | str | No | "TBD" | CIDAR ROUGE-L score |
| --islamic-rouge | str | No | "TBD" | Islamic QA ROUGE-L score |
| --perplexity | str | No | "TBD" | Arabic perplexity |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Fill in model card template with provided scores
4. Write model card to output path
5. Log success

---

## Usage Examples

```bash
# Generate model card with all scores
python -m src.scripts.generate_model_card \
    --output release/Baligh-1.7B-v0-instruct/MODEL_CARD.md \
    --mmlu-score "65.2" \
    --cidar-rouge "42.1" \
    --islamic-rouge "38.5" \
    --perplexity "12.3"

# Generate with TBD scores (fill in later)
python -m src.scripts.generate_model_card \
    --output release/Baligh-1.7B-v0-instruct/MODEL_CARD.md
```

---

## Model Card Template

The generated model card includes:
- Model details (type, base model, parameters, context length, languages)
- Training details (CPT and SFT stages, datasets, methods)
- Evaluation results (MMLU-Arabic, CIDAR, Islamic QA, perplexity)
- Intended use cases
- Limitations and disclaimers
- Citation information
