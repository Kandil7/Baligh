#!/usr/bin/env python3
"""Baseline Qwen3-1.7B on the Sunni-core benchmark (pre-SFT reference point).

Wraps the raw-completion `run_sunni_benchmark` runner with a chat-template
adapter so the instruct model is evaluated the way it will actually be used:
system prompt + user turn, thinking mode disabled, greedy decoding
(benchmarks must be reproducible — same rule as baligh EvalConfig).

Usage:
    uv run python scripts/baseline_sunni_benchmark.py \
        --model Qwen/Qwen3-1.7B \
        --benchmark data/benchmarks/sunni_core_v0.jsonl \
        --output docs/evaluation/sunni_baseline_qwen3_17b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(errors="replace")

SYSTEM = (
    "أنت بليغ، مساعد عربي علمي ملتزم بمنهج أهل السنة والجماعة. "
    "أجب اعتمادًا على المصدر والسياق، وميّز بين النص المنقول والشرح والاستنباط. "
    "لا تنسب آية أو حديثًا أو قولًا إلى الله أو رسوله أو عالم دون توثيق. "
    "عند وجود خلاف معتبر اذكره، ولا تعرض قولًا واحدًا على أنه إجماع. "
    "إذا لم يكف السياق أو لم تثبت المعلومة فقل: لا أعلم أو لا يكفي السياق للجزم. "
    "في الفتوى الشخصية نبّه إلى مراجعة عالم موثوق."
)


class ChatEvaluator:
    """Adapts a chat model to the runner's `.generate(prompt)` interface."""

    def __init__(self, model_name: str, max_new_tokens: int = 512) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        free_gb = __import__("torch").cuda.mem_get_info()[0] / 1e9
        print(f"Loaded {model_name} | free VRAM after load: {free_gb:.1f} GB")

    def generate(self, prompt: str, **_: object) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # fixed mode: fair, reproducible comparison
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--benchmark", default="data/benchmarks/sunni_core_v0.jsonl")
    parser.add_argument("--output", default="docs/evaluation/sunni_baseline_qwen3_17b.json")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    from baligh.evaluation.benchmarks import run_sunni_benchmark

    evaluator = ChatEvaluator(args.model, max_new_tokens=args.max_new_tokens)
    result = run_sunni_benchmark(
        evaluator, benchmark_path=args.benchmark, max_samples=args.max_samples
    )

    report = {
        "model": args.model,
        "benchmark": args.benchmark,
        "decoding": f"greedy, enable_thinking=False, max_new_tokens={args.max_new_tokens}",
        "categories": result["categories"],
        "n_items": result["n_items"],
        "n_auto_scored": result["n_auto_scored"],
        "human_queue_size": len(result["human_queue"]),
        "human_queue": result["human_queue"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out}")
    print(json.dumps({k: v for k, v in result["categories"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
