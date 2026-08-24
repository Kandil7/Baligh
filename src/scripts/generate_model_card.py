"""Generate the HF model card for Baligh-1.7B."""

import argparse
from pathlib import Path

from baligh.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- arabic
- islamic
- qwen3
- continued-pretraining
- instruction-tuning
- qlora
- 1.7b
library_name: transformers
pipeline_tag: text-generation
base_model: unsloth/Qwen3-1.7B-Base
---

# Baligh-1.7B v0

**Baligh-1.7B v0** is an Arabic-first language model with Islamic knowledge specialization, fine-tuned from `unsloth/Qwen3-1.7B-Base`.

## Model Details

- **Model Type**: Causal Language Model (QLoRA adapter + merged instruct variant)
- **Base Model**: unsloth/Qwen3-1.7B-Base (1.7B total params, 1.4B non-embedding)
- **Architecture**: 28 layers, GQA (16 query / 8 KV heads), hidden size 2048
- **Native Context**: 32,768 tokens (architecture); v0 trained at a 2,048-token window - no RoPE scaling applied yet, so reliable context is ~2K until long-context training lands
- **Languages**: Arabic (primary, fusha), English
- **Domains**: General Arabic, Islamic Knowledge

## Training

### Stage 1: Continued Pretraining (CPT)
- **Datasets**: ArabicWeb24 (70%), ArabicText-Large (20%), The Arabic Pile (10%)
- **Islamic Cycle**: Hadith collections, Quran QA, Quran texts
- **Method**: QLoRA 4-bit NF4 (r=16, alpha=16)
- **Sequence Length**: 2048

### Stage 2: Supervised Fine-Tuning (SFT)
- **Datasets**: CIDAR (40%), evol-instruct-arabic (35%), Gazelle (15%), Summarization (10%)
- **Method**: QLoRA 4-bit NF4 (r=16, alpha=16), completion-only loss masking
- **Learning Rate**: 1e-4 (cosine)

## Evaluation

| Benchmark | Score |
|-----------|-------|
| MMLU-Arabic | {mmlu_score} |
| CIDAR ROUGE-L | {cidar_rouge} |
| Islamic QA ROUGE-L | {islamic_rouge} |
| Arabic Perplexity | {perplexity} |

## Intended Use

Baligh-1.7B v0 is designed for:
- Arabic question answering
- Islamic knowledge queries (fiqh, hadith, tafsir)
- Arabic text generation and summarization
- Structured output (JSON extraction)
- Arabic reformulation and translation to fusha

## Limitations

- Not a substitute for qualified Islamic scholars
- May hallucinate on specific religious rulings
- Limited to knowledge cutoff of training data
- Arabic dialect performance varies; v0 targets fusha
- Effective context is the 2K training window, not the 32K native maximum
- Not suitable for high-stakes decisions

## Citation

```bibtex
@software{{baligh2026,
  title={{Baligh-1.7B: Arabic-first LLM with Islamic Knowledge Specialization}},
  author={{Baligh Team}},
  year={{2026}},
  url={{https://huggingface.co/Kandil7/Baligh-1.7B}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Generate model card for Baligh-1.7B")
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

    logger.info(f"Model card generated at {output_path}")


if __name__ == "__main__":
    main()
