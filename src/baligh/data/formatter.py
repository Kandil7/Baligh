"""Prompt formatting for Baligh-1.5B v0."""

from baligh.models.tokenizer import apply_chat_template, format_instruction, get_tokenizer
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


class PromptFormatter:
    def __init__(self, tokenizer_name=None, max_seq_length=2048, packing=False):
        self.tokenizer = get_tokenizer(tokenizer_name, max_seq_length=max_seq_length)
        self.max_seq_length = max_seq_length
        self.packing = packing

    def format_cpt(self, example):
        text = example.get("text", "")
        if not text:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def format_sft(self, example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        if not instruction or not output:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        messages = format_instruction(instruction, input_text, output)
        formatted = apply_chat_template(
            self.tokenizer, messages, tokenize=True, add_generation_prompt=False
        )
        if isinstance(formatted, str):
            tokenized = self.tokenizer(
                formatted,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )
        else:
            tokenized = {"input_ids": formatted, "attention_mask": [1] * len(formatted)}
        labels = tokenized["input_ids"].copy()
        tokenized["labels"] = labels
        return tokenized

    def format_chat(self, messages):
        return apply_chat_template(
            self.tokenizer, messages, tokenize=True, add_generation_prompt=True
        )

    def __call__(self, batch):
        if "instruction" in batch:
            return self.format_sft_batch(batch)
        elif "text" in batch:
            return self.format_cpt_batch(batch)
        return batch

    def format_cpt_batch(self, batch):
        texts = batch["text"]
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for text in texts:
            if not text:
                continue
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            results["labels"].append(tokenized["input_ids"].copy())
        return results

    def format_sft_batch(self, batch):
        instructions = batch.get("instruction", [])
        inputs = batch.get("input", [""] * len(instructions))
        outputs = batch.get("output", [])
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for inst, inp, out in zip(instructions, inputs, outputs, strict=False):
            if not inst or not out:
                continue
            messages = format_instruction(inst, inp, out)
            formatted = apply_chat_template(
                self.tokenizer, messages, tokenize=True, add_generation_prompt=False
            )
            if isinstance(formatted, str):
                tokenized = self.tokenizer(
                    formatted,
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False,
                    return_tensors=None,
                )
            else:
                tokenized = {"input_ids": formatted, "attention_mask": [1] * len(formatted)}
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            results["labels"].append(tokenized["input_ids"].copy())
        return results


def get_cpt_formatter(tokenizer_name=None, max_seq_length=2048, packing=True):
    return PromptFormatter(tokenizer_name, max_seq_length, packing=packing)


def get_sft_formatter(tokenizer_name=None, max_seq_length=2048, packing=False):
    return PromptFormatter(tokenizer_name, max_seq_length, packing=packing)
