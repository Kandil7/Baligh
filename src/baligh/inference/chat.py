"""Chat interface for Baligh-1.7B v0."""

from typing import Any, cast

from baligh.inference.generator import TextGenerator
from baligh.models.tokenizer import apply_chat_template
from baligh.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "أنت بلّاغ، مساعد ذكاء اصطناعي عربي متخصص في المعرفة الإسلامية. أجب بالعربية الفصحى."
)


class ChatBot:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        system_prompt: str | None = None,
        max_history_turns: int = 10,
        max_history_chars: int = 6000,
    ) -> None:
        self.generator = TextGenerator(model_path, adapter_path)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        # Unbounded history re-encodes the whole conversation every turn and
        # eventually overflows the context window; these caps keep prompts
        # bounded (Qwen3-1.7B native window: 32K).
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars
        self.history: list[dict] = []

    def _trimmed_history(self) -> list[dict]:
        """Most recent turns within the configured budget."""
        trimmed = self.history[-self.max_history_turns :]
        total = 0
        start = len(trimmed)
        for i in range(len(trimmed) - 1, -1, -1):
            total += len(trimmed[i]["content"])
            if total > self.max_history_chars:
                start = i + 1
                break
            start = i
        kept = trimmed[start:]
        if len(kept) < len(self.history):
            logger.debug(f"History trimmed: {len(self.history)} -> {len(kept)} turns")
        return kept

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self._trimmed_history()
        rendered = apply_chat_template(
            self.generator.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = cast("str", rendered)
        response = self.generator.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self) -> None:
        self.history = []

    def get_history(self) -> list[dict]:
        return list(self.history)


def chat(
    model_path: str,
    user_message: str,
    adapter_path: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    bot = ChatBot(model_path, adapter_path, system_prompt)
    return bot.chat(user_message, **kwargs)
