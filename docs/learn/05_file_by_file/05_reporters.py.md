# reporters.py - Report Generation

**Path**: `src/baligh/evaluation/reporters.py` (36 lines)

## Purpose

Generate JSON evaluation reports.

---

## generate_eval_report (lines 10-25)

Creates a timestamped JSON report with model name and results dict. Saves to output_dir with filename pattern `eval_report_{model}_{timestamp}.json`.

---

## save_results (lines 27-36)

Generic results saver. Writes any dict to JSON with filename pattern `{prefix}_results_{timestamp}.json`.
