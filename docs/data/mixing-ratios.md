# Dataset Mixing Ratios

## CPT Mixing Ratios

| Dataset | Weight | Rationale |
|---------|--------|-----------|
| ArabicWeb24 | 70% | Largest, highest quality web corpus |
| ArabicText-Large | 20% | High-quality articles, diverse topics |
| The Arabic Pile | 10% | Dialect diversity, MSA + dialects |

Total: 100%

Islamic datasets run as a separate cycle after main CPT.

## SFT Mixing Ratios

| Dataset | Weight | Rationale |
|---------|--------|-----------|
| CIDAR | 40% | Cultural relevance, high-quality Arabic instructions |
| evol-instruct-arabic | 35% | CoT reasoning, multi-turn, diverse tasks |
| islamweb_fatwa | 15% | Domain-specific Islamic knowledge (local fatwa corpus) |
| Summarization | 10% | Compression, key information extraction |

Stage 2 (`configs/sft/sft-stage2.yaml`) shifts weight toward the Islamic
source: CIDAR 35% / evol 30% / islamweb_fatwa 20% / summarization 15%.

Total: 100%

Row budget: `first_exhausted` caps the mix at `min(rows_i / weight_i)`.
Stage 1 binds on islamweb_fatwa (3,525 rows / 0.15 = 23,500 total), so every
fatwa row is consumed and ~6% of the 25,000 ceiling is lost. A dataset too
small for its weight would instead silently cap the WHOLE mix, which is why
Gazelle (a 200-row benchmark) was removed.

## Implementation



## Rationale

- Quality over quantity for v0
- ArabicWeb24 is the cleanest large corpus
- CIDAR provides cultural grounding
- evol-instruct adds reasoning capability
- Islamic QA kept small but present for identity
- Ratios configurable via YAML configs
