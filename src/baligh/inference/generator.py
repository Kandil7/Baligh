"""Text generation for Baligh-1.5B v0."""

import torch
from typing import List, Optional
from baligh.config import get_model_config, get_eval_config
from baligh.models.loader import load_base_model, load_lora_model, merge_lora
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

class TextGenerator:
    def __init__(self, model_path, adapter_path=None, config=None):
        self.config = config or get_eval_config()
        self.model_config = get_model_config()
        self.tokenizer = get_tokenizer(self.model_config.tokenizer_name, self.model_config.max_seq_length)
        
        logger.info("Loading model: %s" % model_path)
        self.model = load_base_model(model_name=model_path, load_in_4bit=True)
        
        if adapter_path:
            logger.info("Loading adapter: %s" % adapter_path)
            self.model = load_lora_model(self.model, adapter_path, is_trainable=False)
        
        self.model.eval()

    def generate(self, prompt, max_new_tokens=None, temperature=None, top_p=None, top_k=None, do_sample=None, repetition_penalty=None, **kwargs):
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        
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
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def generate_batch(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]

def generate(model_path, prompt, adapter_path=None, **kwargs):
    generator = TextGenerator(model_path, adapter_path)
    return generator.generate(prompt, **kwargs)
