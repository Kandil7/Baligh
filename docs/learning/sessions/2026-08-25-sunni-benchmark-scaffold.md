# Sunni-core benchmark v0: seed data + auto-scoring runner

### Context

Baligh repo. Following the SOTA-strategy discussion, priority #1 was building the Sunni-reviewed benchmark (the project's moat). Scaffolded inside the EXISTING evaluation package (lesson from review finding A1 — no parallel pipelines).

### Explanation

Deliverables: (1) `data/benchmarks/sunni_core_v0.jsonl` — 16 seed items across 8 categories (quran_exact_match, hadith_attribution, fiqh_madhhab_aware, aqeedah_sunni, rag_faithfulness, refusal_calibration, arabic_quality, safety_deferral), each with id/question/context/must_cite/must_refuse/required_citation/checker/reference_answer. Only well-established texts used (Ayat al-Kursi 2:255, Bukhari #1, etc.); one trap item (fabricated takhrij+date) and one irrelevant-context RAG item. (2) `run_sunni_benchmark()` in src/baligh/evaluation/benchmarks.py following existing runner conventions (evaluator.generate(prompt) → dict): three checker types — citation (auto: required surah/ayah/hadith-number/book-title fragments must appear verbatim, accepts Eastern-Arabic digits ٠١٢...), refusal (auto: Arabic refusal-marker matching), human (routed to human_queue for the existing HumanEvaluator rubric flow). Returns per-category accuracy + human queue. (3) 7 unit tests in test_benchmarks.py including a trap-fails-citation check. Validation loop worth remembering: first run FAILED because reference answers were behavior descriptions rather than actual passing responses — references must be realistic model outputs or the self-consistency check is meaningless; fixed quran_002/rag_001/safety_002 accordingly. Verified end-to-end with mock evaluator (7 auto-scored, 9 human-queued) + full suite 283 passed @ 82.39% coverage.

### Alternatives

(a) LLM-as-judge scoring — rejected for v0: non-deterministic, costs API money, and for religious content an unsupervised judge is exactly what the methodology forbids. (b) All-human scoring — too slow to bootstrap; hybrid keeps ~44% auto-scored. (c) Separate benchmark module outside baligh.evaluation — rejected per A1 consolidation lesson from the prep-script review.

### Rationale (Why this?)

Auto-checkers are deliberately conservative (pass only on explicit verifiable fragments); they can false-fail stylistically-different correct answers but never false-pass fabrication — right asymmetry for this domain. The benchmark doubles as the validation set philosophy: same items gate both training decisions and release decisions. Revisit if citation fragment-matching proves too brittle on real model outputs (e.g., answers citing 'البقرة 255' without digit forms matched).

### Exercises

1. Expand to 50–100 items/category using Athar passages + scholar review, keeping the schema. 2. Run against base Qwen3-1.7B vs current Baligh checkpoint to get baseline numbers before any SFT. 3. Wire run_sunni_benchmark into run_eval CLI as a --benchmarks sunni choice. 4. Test citation checker against paraphrased citations ('سورة البقرة، الآية ٢٥٥') and tighten if false-fails dominate. 5. Feed human_queue items into HumanEvaluator CSV export for the reviewer workflow.

### Next Steps

Priority #2 from the SOTA plan: fix C1 routing in prepare_baligh_final_qwen3_17b.py so Islamic sources actually reach train.jsonl after review; then baseline eval of Qwen3-1.7B on sunni_core_v0 before any fine-tuning.

---
