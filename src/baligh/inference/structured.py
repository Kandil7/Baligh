"""Structured output for Baligh-1.7B v0."""

import json
import re

from baligh.inference.generator import TextGenerator
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_payload(text: str):
    """Best-effort JSON recovery from a 1.5B-model response.

    Tolerates, in order: pure JSON, markdown-fenced JSON, and the first
    balanced {...} / [...] block embedded in surrounding prose (the common
    real-world failure mode for small models).
    """
    if not text:
        raise ValueError("Empty response — no JSON to extract")

    candidates: list[str] = []

    fenced = _FENCE_RE.findall(text)
    candidates.extend(chunk.strip() for chunk in reversed(fenced))

    stripped = text.strip()
    candidates.append(stripped)

    for opener in ("{", "["):
        start = stripped.find(opener)
        if start != -1:
            close = "}" if opener == "{" else "]"
            depth = 0
            for i in range(start, len(stripped)):
                ch = stripped[i]
                if ch == opener:
                    depth += 1
                elif ch == close:
                    depth -= 1
                    if depth == 0:
                        candidates.append(stripped[start : i + 1])
                        break

    last_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise ValueError(f"No valid JSON found in response: {last_error}")


class StructuredOutput:
    def __init__(self, model_path, adapter_path=None):
        self.generator = TextGenerator(model_path, adapter_path)

    def extract_json(self, prompt, schema=None, max_retries=3, temperature=0.1):
        """Prompt the model for JSON and parse robustly.

        Retries escalate: each failure appends a corrective instruction so
        repeated sampling at identical temperature isn't near-deterministic.
        """
        base_instruction = "\n\nRespond ONLY with valid JSON. No extra text."
        current_prompt = prompt + base_instruction
        for attempt in range(max_retries):
            response = self.generator.generate(
                current_prompt,
                temperature=temperature,
                do_sample=attempt > 0,
                max_new_tokens=1024,
            )
            try:
                data = extract_json_payload(response)
                if schema is not None:
                    logger.debug("Schema provided; validation hook available via pydantic")
                return data
            except ValueError:
                logger.warning(f"Failed to parse JSON, retry {attempt + 1}/{max_retries}")
                current_prompt = (
                    f"{prompt}{base_instruction}\n"
                    f"(Previous attempt was invalid JSON. Output raw JSON only.)"
                )
        raise ValueError(f"Failed to extract valid JSON after {max_retries} retries")


def extract_json(model_path, prompt, adapter_path=None, **kwargs):
    extractor = StructuredOutput(model_path, adapter_path)
    return extractor.extract_json(prompt, **kwargs)
