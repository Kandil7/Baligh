# Review: Baligh plan transcript + prepare_baligh_final_qwen3_17b.py

### Context

Baligh repo (D:\AI\Projects\LLM\Baligh). User requested full detailed review of docs/plan/Baligh assets reviewed.md (exported Perplexity research transcript, 8 sections on base-model choice, dataset landscape, prep/training guides for Baligh-1.7B on Qwen3-1.7B) and docs/plan/prepare_baligh_final_qwen3_17b.py (standalone SFT data prep script). Reviewed against src/baligh/data pipeline (loader/cleaner/formatter/mixer/validators + DATASETS registry) during the same session in which the smoke test exposed three stale registry entries.

### Explanation

VERDICT: Approve with changes. Key findings — SCRIPT: (C1, major) review=(repo != bobez999) at line 109 routes ALL Islamic sources to needs_review.jsonl, so train.jsonl ends up containing only generic Arabic QA — the Islamic core never trains; must be a deliberate per-source policy. (C2, major) validation split groups only by source_dataset → intra-source topical leakage across 73.6K Islamweb rows; need composite group keys (question_number/URL/book_title). (C3) load_repo's broad except doesn't record failed sources in manifest. (C4) Dataset.from_list([]) unguarded. (C5) 'weight' is a post-load multiplicative cap, not the mix percentage the MD describes. (A1, major) script duplicates the existing baligh.data pipeline as a parallel schema — drift risk proven real since the single registry already rotted (google/mc4 and oscar-corpus/oscar_ar deleted upstream, TheArabicPile_Dialects now needs config='dedup'); recommendation: port SOURCES into DATASETS registry and reuse cleaner/formatter instead of maintaining two pipelines. SECURITY: clean (env-only HF token, no pickle, parquet-only ingestion). MD DOC: (M1) exported footnote markers corrupt inline code (output[^9_0] should be output[0]; ds[^4_0] should be ds[0]). (M2) citation reliability low — footnotes often don't support claims; all dataset statistics (73.6K, 17,944, 5,852 chunks) unverified. (M3) internal contradictions: three different mix-ratio recipes and conflicting readiness tiers across sections. (m4) packing=True + assistant_only_loss=True version-sensitive; Qwen3 thinking-mode policy undefined at training time (only at inference). STRENGTHS confirmed: methodology (human Sunni review, no fabricated answers, source-level splits, license caution CC-BY-NC-4.0, benchmark taxonomy) is sound; VRAM guidance matches RTX 5000 16GB reality.

### Alternatives

Considered: (a) keep script standalone as exploration tool — pro: fast iteration; con: schema drift vs baligh.data, double maintenance of source metadata. (b) port into registry+pipeline — pro: single source of truth, reuses tested cleaner/formatter; con: upfront refactor effort, registry needs per-source conversion functions (fatwa_row/chat_row/passage_review logic). (c) middle path: keep script but emit baligh-canonical schema so outputs load directly into existing trainer — cheapest immediate fix.

### Rationale (Why this?)

The C1 finding is decisive: as written, running the full pipeline produces a training set with zero Islamic samples, which defeats the project's purpose — yet it looks like success in the manifest. The A1 recommendation follows from observed fact: registry rot already happened once; two pipelines guarantee recurrence. Revisit if user decides the conservative all-to-review routing IS the intended v1 policy.

### Exercises

1. Run the script with --limit-per-source 1000 and inspect manifest.json rows_ready vs rows_needs_review to see the C1 imbalance concretely. 2. Write a test asserting no shared normalized-question prefix between train and validation to detect C2 leakage. 3. Prototype registering Raniahossam33/Islamweb_part2 in DATASETS with a converter hook and run it through the existing formatter to compare output vs make(). 4. Verify assistant_masks generation for Qwen3 chat template with return_assistant_tokens_mask=True before adopting the MD training config. 5. Probe each MD-claimed dataset (NightPrince, Omar-youssef, Islamweb_part2, bobez999) for real schema/splits to replace unverified stats.

### Next Steps

Decide C1 policy (per-source train-vs-review matrix); then either execute alternative (c) canonical-schema bridge or the full port into baligh.data; finish the interrupted smoke test playbook afterwards since prepare_data fixes landed earlier this session (arabic_pile config=dedup, oscar_ar→wikimedia/wikipedia 20231101.ar, mc4_ar→allenai/c4 ar, IterableDataset.map num_proc/desc/remove_columns fixes).

---
