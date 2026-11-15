# Islamweb_part2 Dataset Summary

**Date:** 2026-11-14
**Status:** Reviewed, not integrated
**Source reviewed in:** `docs/plan/Baligh assets reviewed.md` (Arabic review notes)

## What it is

`Raniahossam33/Islamweb_part2` is a Hugging Face dataset of fatwas scraped
from islamweb.net, republished as a single CSV/Parquet dataset.

FACT — verified against the live dataset page on 2026-11-14:

| Property | Value |
| --- | --- |
| Repo id | `Raniahossam33/Islamweb_part2` |
| Rows | 73,600 (one split: `train`) |
| Format | CSV, auto-converted to Parquet |
| Columns | `Question Number` (int64), `URL` (string), `Question` (string), `Question_Context` (string), `Answer` (string) |
| Observed ranges | `Question` 7-149 chars, `Question_Context` 12-6,260 chars, `Answer` 66-15,200 chars |
| Language | Arabic (all sampled URLs are `/ar/fatwa/<id>`) |
| Domains visible in viewer | fiqh, hadith, aqeedah, inheritance, tajweed, family law |

Every row carries its fatwa URL, so provenance is preserved end to end.

## How it differs from the local `islamweb1` corpus

The local corpus at `D:\AI\Datasets\Custom\islamweb1` is a **different scrape**
and is the one wired into training today (`islamweb_fatwa` in the SFT mix).

| | `islamweb1` (local, in use) | `Islamweb_part2` (HF, not integrated) |
| --- | --- | --- |
| Rows | 46,501 raw, 3,917 after cap/dedupe | 73,600 |
| Files | 50 JSON files keyed by fatwa URL | 1 CSV/Parquet |
| Languages | ar, en, fr, de, es, id (multilingual) | Arabic only |
| Schema | `title, question, answer, lang, related, categories` | `Question Number, URL, Question, Question_Context, Answer` |
| Extra context | none | `Question_Context` (asker's details) |
| Built by | `src/scripts/convert_islamweb.py` | not built |

INFERENCE — the two almost certainly overlap in fatwa IDs, since both are
islamweb.net fatwas. This is unverified: no join has been run.

OPEN QUESTION — how many of the 73.6K part2 fatwa IDs are already present in
the local corpus? Answer it by extracting IDs from `URL` and intersecting with
`fatwa_id` in `D:\AI\datasets\baligh\islamweb_fatwa\*.jsonl` before any merge.

## Usage policy (from the review)

The original review's verdict, unchanged:

| Use | Decision |
| --- | --- |
| SFT on question/answer style | Suitable |
| Training citation and source attribution | Highly suitable |
| Treating answers as settled fact | Not safe |
| Continued pretraining | Not suitable |
| Raw-text-only training | No |
| Training fatwas without review | No |

The reason: answers often present one school's ruling or one scholar's
preferenced view. Baligh must learn to attribute ("as stated in this fatwa")
rather than present every answer as consensus. This is why the local
converter's instruction text asks the model to note where scholars differ.

## Integration plan (not started)

1. **Dedupe first.** Extract fatwa IDs from `URL`, intersect with the local
   corpus, and drop collisions. Do not add 73.6K rows blind — at any sane
   weight they would dominate the mix and rebind `first_exhausted`.
2. **Convert with a second converter function.** Schema differs from the local
   one: map `Question` -> `input`, `Answer` -> `output`, and decide whether
   `Question_Context` goes into `input` (it contains the asker's personal
   details, so most likely it is dropped or summarized).
3. **Split by `Question Number`**, never by row, for the same leakage reason
   documented in `src/scripts/convert_islamweb.py`.
4. **Register as a separate entry** (`islamweb_part2`) rather than merging
   into `islamweb_fatwa`, so its weight can be tuned independently.
5. **Recheck mix math.** Any new source changes which dataset binds
   `first_exhausted`; recompute before committing ratios
   (`docs/data/mixing-ratios.md`).

RECOMMENDATION — hold integration until the current mix trains and evaluates.
The local 3,917-row source already binds stage 1 at 15%, so adding 73.6K rows
now would change the mixture under an untrained baseline and make the first
SFT run uninterpretable.

## Risks

- **Single-source concentration.** Both Islamweb sources come from one site.
  At 15-20% weight the model's Islamic behavior is largely one publisher's
  editorial line.
- **Sensitive personal fiqh.** Sampled rows include intimate family and
  worship questions. Any human-eval rubric must cover these categories
  (`docs/evaluation/human-eval-rubric.md` already stratifies Islamic QA).
- **License.** No explicit license is declared for the underlying fatwa text.
  The registry entry records `See source (Islamweb fatwa pages)`; this is a
  release blocker for redistribution, not for internal training.

## Related

- Local converter: `src/scripts/convert_islamweb.py`
- Registry entry: `islamweb_fatwa` in `src/baligh/data/datasets.py`
- Mix ratios and row-budget math: `docs/data/mixing-ratios.md`
- Original Arabic review: `docs/plan/Baligh assets reviewed.md`
