# Session 2026-11-14 — Islamweb fatwa integration into the SFT pipeline

## Goal

Make the local Islamweb fatwa corpus (`D:\AI\Datasets\Custom\islamweb1`)
loadable by Baligh's SFT pipeline, and move the Baligh-1.7B v0 plan forward.

## What was built

- `src/scripts/convert_islamweb.py` — converts 46,501 raw fatwa records to
  the canonical `{instruction, input, output}` schema. Caps non-Arabic rows at
  3,000 seeded-sampled (all 946 Arabic rows kept), dedupes by `fatwa_id`
  preferring Arabic, and splits train/eval by group so one ruling never
  straddles the boundary. Output: 3,917 rows (3,525 train / 392 eval),
  `group_overlap: 0`, `repeated_fatwa_ids: 0`, leakage **PASS**. Re-running
  produces byte-identical artifacts (verified by SHA-256).
- `loader.py` — local JSON/JSONL sources: a split mapping is always local, a
  string is local only if it has a JSON suffix and exists on disk (so Hub ids
  are never mistaken for files).
- `datasets.py` — registered `islamweb_fatwa`, removed `gazelle`, repaired
  the summarization repo id, added a ShareGPT standardizer branch for
  `evol_instruct_arabic`, and made both branches drop every source column so
  the mix gets an exact three-column schema.

- `run_eval --benchmarks sunni` wired up. `run_sunni_benchmark` existed in
  `src/baligh/evaluation/benchmarks.py:196` but the CLI only offered
  `mmlu|cidar|islamic`, so the Sunni benchmark was unreachable. Added the
  `sunni` choice plus `--sunni-path`, and moved both benchmark validations
  (islamic data, sunni file) ahead of model loading so a usage error fails in
  milliseconds instead of after a 1.7B load.

## Defects found and fixed

1. **Gazelle was dead weight.** `Gazelle/arabic-writing` 404s; the real repo
   (`UBC-NLP/gazelle_benchmark`) is 200 rows, and at 0.15 weight
   `first_exhausted` would have capped the entire SFT set at ~1,300 rows.
   Its slot went to `islamweb_fatwa` (straight swap, other ratios untouched).
2. **Summarization repo id was wrong** — `arabic-msa-summarization` does not
   exist; now `Arabic-Synthetic-Summarization-Dataset`.
3. **The Arabic-only cleaning gate destroyed non-Arabic SFT rows.**
   `cleaner.sanitize()` blanks (does not drop) text failing
   `is_arabic(threshold=0.1)`, so with `--clean` it removed the output of
   2,668 of 3,525 fatwa rows and silently damaged cidar/evol/summarization.
   Fixed by giving the SFT path `get_cleaning_pipeline(language_filter=None)`
   while CPT keeps the gate. Verified: islamweb now 200/200 valid in the dry
   run; the 3 remaining invalid rows are pre-existing in the raw sources
   (confirmed identical before and after cleaning).
4. **Source columns leaked into the mix.** `datasets.map(remove_columns=...)`
   affects the output only — the mapper still reads full input — so the
   column-based standardizer's keep-list let `text`/`summary` survive and
   handed `interleave_datasets` a union schema. Both branches now pass
   `remove_columns=list(dataset.column_names)`.
5. **CI was red before this session** (pre-existing): `black` wanted
   `models/quantization.py` reformatted, `mypy` flagged
   `evaluation/metrics.py:119` returning `Any` as `float`. Both fixed.

## Verification evidence

- `pytest tests` — **292 passed**, coverage **82.5%** (gate 80%), exit 0.
- `ruff check src/` clean, `black --check src/` clean (45 files),
  `mypy src/` clean (49 files).
- SFT dry run `--clean --max-samples 200` — exit 0,
  `Saved sft: {'train': 496, 'eval': 11}`.
- All four SFT sources load and standardize: cidar 10,000, evol 59,022,
  islamweb_fatwa 3,525, summarization 4,000 — each to exactly
  `{instruction, input, output}`.
- Mix: 23,239 rows with `strict=True`; stage-1 budget binds on islamweb at
  23,500 predicted (islamweb fully consumed, 6% of ceiling lost).
- `fatwa_id` overlap between train and eval: 0.
- `LICENSE` added and diffed against `apache.org/licenses/LICENSE-2.0.txt` —
  byte-identical apart from the copyright line.

## Docs synced

`README.md`, `docs/data/mixing-ratios.md`, `docs/data/datasets-catalog.md`,
`docs/architecture/data-flow.md`, `docs/training/sft-config.md`,
`prompt/01-create-preference-data.md`, `AGENTS.md` (base model now matches
the ADR), `generate_model_card.py` (mix now derived from live config instead
of a hardcoded string).

New: `docs/data/islamweb_part2_dataset_summary.md`,
`docs/plan/qwen3_1.7b_baligh_starter_plan.md`.

Historical records (`docs/learning/**`, `docs/plan/عاوز اعمل Baligh-1.5B v0.md`)
were deliberately left unedited.

## Blocked / next

- venv torch is CPU-only (`2.13.0+cpu`) while `uv.lock` pins cu128; `uv sync`
  not yet run (needs approval, ~2.75 GB).
- Playbook `prompt/07-smoke-test.md` not started; `training/smoke_*` absent.
- No baseline `run_eval --benchmarks sunni`, no training runs, no checkpoints.

## Gotchas worth keeping

- `remove_columns` in `datasets.map` affects only the output columns; the
  mapper still sees the full input row.
- `get_sft_config()` returns a frozen default — editing only the YAML does
  not change the trained mixture.
- Windows console encoding: set `PYTHONIOENCODING=utf-8` before printing
  Arabic, and use `.venv\Scripts\python.exe` directly (uv re-resolution can
  time out).
- `validate_dataset` returns `(valid, invalid, errors_dict)` — `errors` is a
  dict, not a list.
