"""Structured output for Baligh-1.5B v0."""

import json
import re
from typing import Dict, Any, Optional
from baligh.inference.generator import TextGenerator
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

class StructuredOutput:
    def __init__(self, model_path, adapter_path=None):
        self.generator = TextGenerator(model_path, adapter_path)

    def extract_json(self, prompt, schema=None, max_retries=3):
        json_prompt = prompt + "

Respond ONLY with valid JSON. No extra text."
        for attempt in range(max_retries):
            response = self.generator.generate(json_prompt, temperature=0.1, max_new_tokens=1024)
            try:
                data = json.loads(response)
                if schema:
                    pass  # Schema validation placeholder
                return data
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON, retry %d/%d" % (attempt + 1, max_retries))
        raise ValueError("Failed to extract valid JSON after %d retries" % max_retries)

def extract_json(model_path extract_json(model_path, prompt, adapter_path=None, **kwargs):
    extractor = StructuredOutput(model_path, adapter_path)
    return extractor.extract_json(prompt, **kwargs)
