"""Inference package for Baligh-1.5B v0."""

from baligh.inference.chat import ChatBot, chat
from baligh.inference.generator import TextGenerator, generate
from baligh.inference.structured import StructuredOutput, extract_json

__all__ = [
    "TextGenerator",
    "generate",
    "ChatBot",
    "chat",
    "StructuredOutput",
    "extract_json",
]
