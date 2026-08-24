"""Main evaluator for Baligh-1.7B v0."""

import torch
from tqdm import tqdm

from baligh.config import get_eval_config, get_model_config
from baligh.models.loader import load_model_with_adapter
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger
from baligh.utils.memory import clear_memory, log_memory_stats

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model_path, adapter_path=None, config=None):
        self.config = config or get_eval_config()
        self.model_config = get_model_config()
        self.tokenizer = get_tokenizer(
            self.model_config.tokenizer_name, self.model_config.max_seq_length
        )

        logger.info(f"Loading model for evaluation: {model_path}")
        # is_inference=True: KV cache on, no k-bit training prep.
        self.model = load_model_with_adapter(
            model_path_or_name=model_path,
            adapter_path=adapter_path,
            is_inference=True,
        )
        self.model.eval()
        log_memory_stats(prefix="After model load")

    def generate(
        self, prompt, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None
    ):
        # `is not None` guards: `or` would turn explicit temperature=0
        # (greedy) into the config default - making deterministic decoding
        # impossible through this API.
        max_new_tokens = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = self.config.temperature if temperature is None else temperature
        top_p = self.config.top_p if top_p is None else top_p
        top_k = self.config.top_k if top_k is None else top_k
        do_sample = self.config.do_sample if do_sample is None else do_sample

        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.config.repetition_penalty,
        )
        if do_sample:
            # Sampling-only params; passing them with do_sample=False warns.
            generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response

    def evaluate_dataset(self, dataset, prompt_template=None, max_samples=None):
        results = []
        total = len(dataset)
        limit = min(max_samples or total, total)

        pbar = tqdm(dataset.select(range(limit)), desc="Evaluating", total=limit)
        for i, example in enumerate(pbar):
            if prompt_template:
                prompt = prompt_template.format(**example)
            else:
                prompt = example.get("prompt", example.get("instruction", ""))
            response = self.generate(prompt)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "reference": example.get("output", example.get("answer", "")),
                    "metadata": example,
                }
            )
            if i % 100 == 0:
                clear_memory()
        return results


def evaluate_model(model_path, adapter_path=None, dataset=None, config=None):
    evaluator = Evaluator(model_path, adapter_path, config)
    if dataset:
        return evaluator.evaluate_dataset(dataset)
    return evaluator
