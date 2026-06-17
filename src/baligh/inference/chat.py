"""Chat interface for Baligh-1.5B v0."""

from baligh.inference.generator import TextGenerator
from baligh.models.tokenizer import apply_chat_template
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


class ChatBot:
    def __init__(self, model_path, adapter_path=None, system_prompt=None):
        self.generator = TextGenerator(model_path, adapter_path)
        self.system_prompt = (
            system_prompt
            or "You are Baligh, an Arabic-first AI assistant with Islamic knowledge. Respond in formal Arabic (fusha)."
        )
        self.history = []

    def chat(self, user_message, max_new_tokens=512, temperature=0.7):
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        prompt = apply_chat_template(
            self.generator.tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        response = self.generator.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.history = []

    def get_history(self):
        return self.history


def chat(model_path, user_message, adapter_path=None, system_prompt=None, **kwargs):
    bot = ChatBot(model_path, adapter_path, system_prompt)
    return bot.chat(user_message, **kwargs)
