"""Prompt formatting for Baligh-1.7B v0.

Contracts:
- ``format_cpt`` / ``format_sft`` process ONE example and return lists.
- ``__call__`` processes a BATCH dict of columns (for ``dataset.map(batched=True)``).

SFT loss masking: when ``response_only_loss`` is enabled, labels are -100 on
the prompt span (system+user+generation prompt) and token ids on the
assistant response. Implemented by tokenizing the prompt-only conversation
and masking that prefix length — no brittle string matching on rendered
special tokens.

All text rendering goes through one path: chat template to string, then a
single tokenizer call with explicit truncation. Long examples can never
slip past max_seq_length.
"""

from baligh.models.tokenizer import apply_chat_template, format_instruction, get_tokenizer
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100

EMPTY_RESULT: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}


class PromptFormatter:
    def __init__(
        self,
        tokenizer_name: str | None = None,
        max_seq_length: int = 2048,
        packing: bool = False,
        response_only_loss: bool = True,
    ) -> None:
        self.tokenizer = get_tokenizer(tokenizer_name, max_seq_length=max_seq_length)
        self.max_seq_length = max_seq_length
        self.packing = packing
        self.response_only_loss = response_only_loss

    # ------------------------------------------------------------------ CPT

    def format_cpt(self, example: dict) -> dict:
        """Tokenize one raw-text example for language modeling."""
        text = example.get("text") or ""
        if not text:
            return dict(EMPTY_RESULT)
        tokenized = self._tokenize(text)
        if tokenized is None:
            return dict(EMPTY_RESULT)
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    def format_cpt_batch(self, batch: dict) -> dict:
        results: dict[str, list]
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        for text in batch["text"]:
            item = self.format_cpt({"text": text})
            if item["input_ids"]:
                results["input_ids"].append(item["input_ids"])
                results["attention_mask"].append(item["attention_mask"])
                results["labels"].append(item["labels"])
        return results

    # ------------------------------------------------------------------ SFT

    def format_sft(self, example: dict) -> dict:
        """Format one instruction example, optionally masking the prompt.

        Falls back gracefully when callers pass pre-standardized examples
        that use different column names via ``columns`` mapping upstream
        (see baligh.data.datasets.standardize_sft_dataset).
        """
        instruction = example.get("instruction") or ""
        output = example.get("output") or ""
        if not instruction or not output:
            return dict(EMPTY_RESULT)
        input_text = example.get("input") or ""

        # Full conversation (prompt + response)
        messages = format_instruction(instruction, input_text, output)
        full_text = self._render(messages, add_generation_prompt=False)

        if not self.response_only_loss:
            tokenized = self._tokenize(full_text)
            if tokenized is None:
                return dict(EMPTY_RESULT)
            tokenized["labels"] = list(tokenized["input_ids"])
            return tokenized

        # Prompt-only conversation (user turn + generation prompt)
        prompt_messages = format_instruction(instruction, input_text, output=None)
        prompt_text = self._render(prompt_messages, add_generation_prompt=True)

        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_length, padding=False
        )["input_ids"]
        full = self._tokenize(full_text)
        if full is None:
            return dict(EMPTY_RESULT)

        n_prompt = len(prompt_ids)
        if n_prompt >= len(full["input_ids"]):
            # Prompt consumed (or exceeded) the whole window: nothing left to
            # train on. Skip rather than emit an all-masked example.
            return dict(EMPTY_RESULT)

        labels = [IGNORE_INDEX] * n_prompt + list(full["input_ids"][n_prompt:])
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    def format_sft_batch(self, batch: dict) -> dict:
        results: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
        instructions = batch.get("instruction", [])
        inputs = batch.get("input", [""] * len(instructions))
        outputs = batch.get("output", [])
        for inst, inp, out in zip(instructions, inputs, outputs, strict=False):
            item = self.format_sft({"instruction": inst, "input": inp or "", "output": out or ""})
            if item["input_ids"]:
                results["input_ids"].append(item["input_ids"])
                results["attention_mask"].append(item["attention_mask"])
                results["labels"].append(item["labels"])
        return results

    # ------------------------------------------------------------- dispatch

    def __call__(self, batch: dict) -> dict:
        if "instruction" in batch:
            return self.format_sft_batch(batch)
        if "text" in batch:
            return self.format_cpt_batch(batch)
        return batch

    # -------------------------------------------------------------- helpers

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        """Render a conversation to a string (never to token ids)."""
        rendered = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, str):  # defensive: tokenizer ignored tokenize=False
            rendered = self.tokenizer.decode(rendered)
        return rendered

    def _tokenize(self, text: str) -> dict | None:
        """Tokenize with enforced truncation. Returns None if nothing survives."""
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if not tokenized["input_ids"]:
            return None
        return {
            "input_ids": list(tokenized["input_ids"]),
            "attention_mask": list(tokenized["attention_mask"]),
        }

    def format_chat(self, messages: list[dict]) -> str | list[int]:
        return apply_chat_template(
            self.tokenizer, messages, tokenize=True, add_generation_prompt=True
        )


def get_cpt_formatter(
    tokenizer_name: str | None = None, max_seq_length: int = 2048, packing: bool = True
) -> PromptFormatter:
    return PromptFormatter(tokenizer_name, max_seq_length, packing=packing)


def get_sft_formatter(
    tokenizer_name: str | None = None,
    max_seq_length: int = 2048,
    packing: bool = False,
    response_only_loss: bool = True,
) -> PromptFormatter:
    return PromptFormatter(
        tokenizer_name,
        max_seq_length,
        packing=packing,
        response_only_loss=response_only_loss,
    )
