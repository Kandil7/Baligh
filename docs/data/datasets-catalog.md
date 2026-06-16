# Dataset Catalog

## CPT Datasets

### ArabicWeb24
- HF: lightonai/ArabicWeb24
- Tokens: ~28B
- Domain: Web crawl (cleaned)
- License: Open
- Streaming: Yes
- Priority: 1 (best quality)

### ArabicText-Large
- HF: Jr23xd23/ArabicText-Large
- Articles: 743K, Words: 244M
- Domain: General Arabic text
- License: Apache 2.0
- Streaming: No
- Priority: 2

### The Arabic Pile (Dialects)
- HF: premio-ai/TheArabicPile_Dialects
- Subsets: 13 (MSA + dialects)
- Domain: Mixed
- License: Open
- Streaming: Yes
- Priority: 3

## Islamic Datasets

### hadith_datasets
- HF: meeAtif/hadith_datasets
- Content: 6 major books
- Format: JSON + CSV
- License: Open

### quran-question-answer-context
- HF: nazimali/quran-question-answer-context
- Content: Quran QA
- Format: JSON
- License: CC

### quran_md
- HF: Buraaq/quran-audio-text-dataset
- Content: Verse-level Arabic
- Format: Multi
- License: Open

## SFT Datasets

### CIDAR
- HF: arbml/CIDAR
- Size: 10K instructions
- Style: Arabic culturally relevant
- License: Apache 2.0
- Columns: instruction, input, output
- Priority: 1 (40%)

### evol-instruct-arabic
- HF: FreedomIntelligence/evol-instruct-arabic
- Size: 50K+
- Style: CoT + multi-turn
- License: Open
- Columns: instruction, input, output
- Priority: 2 (35%)

### Gazelle
- HF: Gazelle/arabic-writing
- Style: Writing assistance
- License: Open
- Columns: instruction, input, output
- Priority: 3 (10%)

### Summarization
- HF: BounharAbdelaziz/arabic-msa-summarization
- Style: Arabic summarization
- License: Open
- Columns: text, summary
- Priority: 4 (10%)

## Evaluation Datasets

### MMLU-Arabic
- HF: FreedomIntelligence/MMLU_Arabic
- Purpose: Arabic QA/Benchmark

### CIDAR-EVAL-100 / CIDAR-MCQ-100
- HF: arbml/CIDAR
- Purpose: Cultural relevance eval

### mr-tydi Arabic
- HF: castorini/mr-tydi (arabic)
- Purpose: Retrieval QA
