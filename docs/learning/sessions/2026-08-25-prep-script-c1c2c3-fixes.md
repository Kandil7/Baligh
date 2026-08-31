# Fix C1/C2/C3 in prepare_baligh_final_qwen3_17b.py + real-data smoke

### Context

Baligh repo. Continuing SOTA sequence: fixing the three major findings from the prep-script review in docs/plan/prepare_baligh_final_qwen3_17b.py, then validating against real HF data.

### Explanation

C1 FIX: replaced inline `review=(repo != bobez999)` with an explicit per-source policy matrix — SOURCES is now a list of dicts each carrying policy: 'ready'|'review'. Policy: NightPrince + Omar-youssef (instruction-formatted QA) = ready → actually reach train.jsonl now; Islamweb fatwas + Athar-RAG-Hub passages = review. A repeatable `--promote SOURCE_ID` CLI flag moves any review source into train while manifest.promoted_to_train records it — routing decisions are always explicit and auditable. needs_sunni_review is stamped centrally in build(); row functions no longer decide routing. C2 FIX: leakage_group_key() composes source_dataset + question_number / source_url / citation instead of source-only grouping; plus fixed a discovered edge bug where n=max(1,...) let singleton groups vanish entirely into validation (now singletons stay in train). C3 FIX: load_repo returns (items, failed); failures land in manifest.sources_failed and are appended to manifest.warning. Also made stdout encoding-safe for cp1252 Windows consoles (sys.stdout.reconfigure(errors='replace')) — artifacts were always utf-8, only console print crashed. VALIDATION: dry-run with monkeypatched loader verified promote/failure/composite-group logic (caught two of my own fixture bugs: identical content for two sources correctly deduped; wrong expected group count since review rows never enter the split). REAL smoke (--limit-per-source 30): all 5 sources loaded successfully — FACT contradicting the MD's claim that Athar-RAG-Hub/Mini-v2 fail to load — yielding 83 rows: 43 ready → train(39)/validation(4), 40 → needs_review. Notable: bobez999 yields few usable rows (~4/30) due to short-answer filters.

### Alternatives

(a) Promote all Islamic QA sources by default — rejected: violates the methodology's own review gate for fatwas/passages. (b) Keep everything in review like the original code but document it — rejected: trains Baligh on zero Islamic data (the original C1 defect). (c) Three-state policy with 'spot-audit' tier — deferred; current binary + promote flag covers v1.

### Rationale (Why this?)

The policy matrix makes the train-vs-review decision a visible config line rather than buried comparison logic; the promote flag keeps human judgment in the loop with manifest traceability. Revisit when the Sunni review workflow lands — promoted sources should flip back to review until reviewed batches exist.

### Exercises

1. Full run without --limit-per-source into data/baligh_final_qwen3_1.7b and inspect source_counts vs the MD's claimed dataset sizes. 2. Spot-audit 50 random NightPrince rows for Sunni-methodology fit; if pass rate <90% flip its policy back to review. 3. Add unit tests for leakage_group_key precedence order. 4. Investigate why bobez999 rows mostly fail the length filter (schema mismatch or genuinely short answers?). 5. Wire this script's outputs into src/baligh SFT trainer as an alternative data path (bridge toward the A1 consolidation).

### Next Steps

SOTA sequence continues: baseline Qwen3-1.7B on sunni_core_v0 benchmark before any SFT; then DPO preference data generation targeting refusal/citation failure modes measured by that baseline.

---
