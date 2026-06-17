# run_eval.py - Evaluation Script

**Path**: `src/scripts/run_eval.py` (47 lines)

## Purpose

CLI entry point for running evaluation on the Baligh model across multiple benchmarks.

---

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| --model-path | str | Yes | - | Path to model |
| --adapter-path | str | No | None | Path to LoRA adapter |
| --output-dir | str | No | "eval/results" | Output directory |
| --benchmarks | nargs+ | No | ["mmlu", "cidar", "islamic"] | Benchmarks to run |
| --max-samples | int | No | 100 | Max samples per benchmark |

---

## Flow

1. Parse command-line arguments
2. Setup logging
3. Load model with Evaluator class
4. Run selected benchmarks
5. Generate evaluation report
6. Log completion

---

## Usage Examples

```bash
# Run all benchmarks
python -m src.scripts.run_eval --model-path training/sft/final

# Run specific benchmarks
python -m src.scripts.run_eval --model-path training/sft/final --benchmarks mmlu cidar

# With LoRA adapter
python -m src.scripts.run_eval --model-path training/cpt/final --adapter-path training/sft/final

# Limit samples for quick testing
python -m src.scripts.run_eval --model-path training/sft/final --max-samples 10
```
