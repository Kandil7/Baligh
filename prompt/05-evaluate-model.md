---
description: Evaluate the final model (after DPO) on Arabic benchmarks and produce an evaluation report.
agent: model-evaluator
model: ollama/qwen3.5:9b
---

# Goal

Evaluate the DPO-aligned model on Arabic benchmarks and produce a comparison report against baseline models.

# Pre-requisites

- `training/dpo/final/` must exist (run prompt `04-run-dpo` first).

# Steps to Execute

1. Read `src/baligh/evaluation/` to understand the evaluation framework.
2. Read `configs/eval/eval-config.yaml` for generation parameters.
3. Run evaluation on all benchmarks:
   ```bash
   uv run python -m src.scripts.run_eval \
     --model-path training/dpo/final \
     --benchmarks mmlu cidar islamic \
     --output-dir docs/evaluation
   ```
4. If the model has a LoRA adapter (not merged), pass the adapter path:
   ```bash
   uv run python -m src.scripts.run_eval \
     --model-path training/sft/final \
     --adapter-path training/dpo/final \
     --benchmarks mmlu cidar islamic \
     --output-dir docs/evaluation
   ```
5. Read the evaluation results from `docs/evaluation/`.
6. Create a comparison table in `docs/evaluation/sota-report.md`:

   | Model | MMLU-Arabic | CIDAR ROUGE-L | Islamic QA | Params |
   |-------|-------------|---------------|------------|--------|
   | Baligh-1.7B v0 | X | X | X | 1.7B |
   | Qwen3-1.7B-Instruct | baseline | baseline | baseline | 1.7B |
   | ALLaM-7B | baseline | baseline | baseline | 7B |
   | Falcon-H1-Arabic-7B | baseline | baseline | baseline | 7B |

7. Run human evaluation on 50–100 samples using the rubric in `configs/eval/eval-config.yaml`:
   - Correctness (1-5)
   - Clarity (1-5)
   - Arabic quality (1-5)
   - Usefulness (1-5)
   - Faithfulness (1-5)

# Constraints

- Do NOT fabricate benchmark scores — only report measured values.
- Do NOT run evaluation on the training split.
- Keep the evaluation script and configs used alongside the report.
- If any benchmark fails, report the error and partial results.
