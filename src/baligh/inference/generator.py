"""Text generation for Baligh-1.7B v0."""

import torch

from baligh.config import get_eval_config, get_model_config
from baligh.models.loader import load_model_with_adapter
from baligh.models.tokenizer import get_tokenizer
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


class TextGenerator:
    def __init__(self, model_path, adapter_path=None, config=None):
        self.config = config or get_eval_config()
        self.model_config = get_model_config()
        self.tokenizer = get_tokenizer(
            self.model_config.tokenizer_name, self.model_config.max_seq_length
        )

        logger.info(f"Loading model: {model_path}")
        # is_inference=True keeps the KV cache enabled and skips k-bit
        # training preparation (which would disable it).
        self.model = load_model_with_adapter(
            model_path_or_name=model_path,
            adapter_path=adapter_path,
            is_inference=True,
        )
        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        do_sample=None,
        repetition_penalty=None,
    ):
        """Generate a completion for *prompt*.

        All overrides use ``is None`` semantics: passing temperature=0 means
        greedy, not "fall back to config".
        """
        max_new_tokens = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = self.config.temperature if temperature is None else temperature
        top_p = self.config.top_p if top_p is None else top_p
        top_k = self.config.top_k if top_k is None else top_k
        do_sample = self.config.do_sample if do_sample is None else do_sample
        repetition_penalty = (
            self.config.repetition_penalty if repetition_penalty is None else repetition_penalty
        )

        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response

    def generate_batch(self, prompts, **kwargs):
        return [self.generate(p, **kwargs) for p in prompts]


def generate(model_path, prompt, adapter_path=None, **kwargs):
    generator = TextGenerator(model_path, adapter_path)
    return generator.generate(prompt, **kwargs)
