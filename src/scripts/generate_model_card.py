MODEL_CARD_TEMPLATE = """
---
license: apache-2.0
tags:
- arabic
- islamic
- qwen2.5
- continued-pretraining
- instruction-tuning
- 1.5b
library_name: transformers
pipeline_tag: text-generation
---

# Baligh-1.5B v0

**Baligh-1.5B v0** is an Arabic-first language model with Islamic knowledge specialization, built on Qwen2.5-1.5B Base.

## Model Details

- **Model Type**: Causal Language Model
- **Base Model**: Qwen/Qwen2.5-1.5B
- **Parameters**: 1.54B
- **Context Length**: 32,768 tokens
- **Languages**: Arabic (primary), English
- **Domains**: General Arabic, Islamic Knowledge

## Training

### Stage 1: Continued Pretraining (CPT)
- **Datasets**: ArabicWeb24 (70%), ArabicText-Large (20%), The Arabic Pile (10%)
- **Islamic Cycle**: Hadith, Quran QA, Quran texts
- **Tokens**: ~20-50B Arabic tokens
- **Method**: QLoRA 4-bit (r=16, alpha=16)
- **Sequence Length**: 2048

### Stage 2: Supervised Fine-Tuning (SFT)
- **Datasets**: CIDAR (40%), evol-instruct-arabic (35%), Gazelle (10%), Summarization (10%), Islamic QA (5%)
- **Examples**: ~100K-500K instruction examples
- **Method**: QLoRA 4-bit (r=16, alpha=16)
- **Learning Rate**: 1e-4

## Evaluation

| Benchmark | Score |
|-----------|-------|
| MMLU-Arabic | {mmlu_score} |
| CIDAR ROUGE-L | {cidar_rouge} |
| Islamic QA ROUGE-L | {islamic_rouge} |
| Arabic Perplexity | {perplexity} |

## Intended Use

Baligh-1.5B v0 is designed for:
- Arabic question answering
- Islamic knowledge queries (fiqh, hadith, tafsir)
- Arabic text generation and summarization
- Structured output (JSON extraction)
- Arabic reformulation and translation to fusha

## Limitations

- Not a substitute for qualified Islamic scholars
- May hallucinate on specific religious rulings
- Limited to knowledge cutoff of training data
- Arabic dialect performance varies
- Not suitable for high-stakes decisions

## Citation


"""

import argparse
from pathlib import Path

from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate model card for Baligh-1.5B")
    parser.add_argument("--output", type=str, required=True, help="Output path for model card")
    parser.add_argument("--mmlu-score", type=str, default="TBD", help="MMLU-Arabic score")
    parser.add_argument("--cidar-rouge", type=str, default="TBD", help="CIDAR ROUGE-L score")
    parser.add_argument("--islamic-rouge", type=str, default="TBD", help="Islamic QA ROUGE-L score")
    parser.add_argument("--perplexity", type=str, default="TBD", help="Arabic perplexity")
    args = parser.parse_args()

    setup_logging()

    card = MODEL_CARD_TEMPLATE.format(
        mmlu_score=args.mmlu_score,
        cidar_rouge=args.cidar_rouge,
        islamic_rouge=args.islamic_rouge,
        perplexity=args.perplexity,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card, encoding="utf-8")

    logger.info("Model card generated at %s" % output_path)


if __name__ == "__main__":
    main()
