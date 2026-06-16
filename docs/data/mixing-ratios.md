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
| Gazelle | 10% | Writing assistance, style transfer |
| Summarization | 10% | Compression, key information extraction |
| Islamic QA | 5% | Domain-specific Islamic knowledge |

Total: 100%

## Implementation



## Rationale

- Quality over quantity for v0
- ArabicWeb24 is the cleanest large corpus
- CIDAR provides cultural grounding
- evol-instruct adds reasoning capability
- Islamic QA kept small but present for identity
- Ratios configurable via YAML configs
