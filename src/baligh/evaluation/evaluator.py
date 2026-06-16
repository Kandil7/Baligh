"""Main evaluator for Baligh-1.5B v0."""

import torch
from tqdm import tqdm
from baligh.config import get_eval_config, get_model_config
from baligh.models.loader import load_base_model, load_lora_model, merge_lora
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger
from baligh.utils.memory import log_memory_stats, clear_memory

logger = get_logger(__name__)

class Evaluator:
    def __init__(self, model_path, adapter_path=None, config=None):
        self.config = config or get_eval_config()
        self.model_config = get_model_config()
        self.tokenizer = get_tokenizer(self.model_config.tokenizer_name, self.model_config.max_seq_length)
        
        logger.info("Loading model for evaluation: %s" % model_path)
        self.model = load_base_model(model_name=model_path, load_in_4bit=True)
        
        if adapter_path:
            logger.info("Loading LoRA adapter: %s" % adapter_path)
            self.model = load_lora_model(self.model, adapter_path, is_trainable=False)
        
        self.model.eval()
        log_memory_stats(prefix="After model load")

    def generate(self, prompt, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None):
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.repetition_penalty,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def evaluate_dataset(self, dataset, prompt_template=None, max_samples=None):
        results = []
        max_samples = max_samples or len(dataset)
        for i, example in enumerate(tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc="Evaluating")):
            if prompt_template:
                prompt = prompt_template.format(**example)
            else:
                prompt = example.get("prompt", example.get("instruction", ""))
            response = self.generate(prompt)
            results.append({
                "prompt": prompt,
                "response": response,
                "reference": example.get("output", example.get("answer", "")),
                "metadata": example
            })
            if i % 100 == 0:
                clear_memory()
        return results

def evaluate_model(model_path, adapter_path=None, dataset=None, config=None):
    evaluator = Evaluator(model_path, adapter_path, config)
    if dataset:
        return evaluator.evaluate_dataset(dataset)
    return evaluator
