# cleaner.py - Text Cleaning

**Path**: `src/baligh/data/cleaner.py` (binary/cannot read directly)

## Purpose

HTML/URL removal, Arabic normalization, and quality filters for text cleaning.

---

## Overview

The cleaner.py file implements the CleaningPipeline class that chains multiple text transformations for Arabic text preprocessing. The pipeline typically includes:

1. **HTML tag removal**: Strip HTML tags from web-scraped text
2. **URL removal**: Remove http/https patterns
3. **Arabic normalization**:
   - Alef variants (alef with hamza, alef with madda) to plain alef
   - Yaa variants (tail yaa) to standard yaa
   - Tatweel (elongation character) removed
   - Tashkeel (diacritics) optionally stripped
4. **Whitespace normalization**: Collapse multiple spaces/newlines
5. **Quality filters**: Min/max length, language detection

---

## Expected Interface

Based on usage in prepare_data.py:

```python
from baligh.data import get_cleaning_pipeline, deduplicate_dataset

# Get the cleaning pipeline
pipeline = get_cleaning_pipeline()

# Clean a batch of texts
cleaned_texts = pipeline.clean_batch(texts)

# Deduplicate a dataset
deduplicated = deduplicate_dataset(dataset, text_column="text")
```

---

## Arabic Normalization Details

The cleaning pipeline handles Arabic-specific normalization:

1. **Alef normalization**: Maps all alef variants to plain alef
   - alef with hamza (U+0623) to alef (U+0627)
   - alef with madda (U+0622) to alef (U+0627)
   - alef with hamza below (U+0625) to alef (U+0627)

2. **Yaa normalization**: Maps tail yaa (U+0649) to standard yaa (U+064A)

3. **Tatweel removal**: Removes elongation character (U+0640)

4. **Tashkeel stripping**: Removes diacritical marks (optional)
   - fatha (U+064E)
   - damma (U+064F)
   - kasra (U+0650)
   - sukun (U+0652)
   - shadda (U+0651)

5. **Whitespace normalization**: Collapses multiple spaces, tabs, newlines to single space
