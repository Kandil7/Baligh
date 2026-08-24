"""Tokenizer utilities for Baligh-1.7B v0."""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from baligh.config import get_model_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def get_tokenizer(
    tokenizer_name: str | None = None,
    max_seq_length: int = 2048,
    padding_side: str = "right",
    truncation_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Get configured tokenizer.

    This is the single tokenizer factory; ``baligh.models.loader`` delegates
    here so behavior cannot drift between modules.

    Args:
        tokenizer_name: Tokenizer name or path.
        max_seq_length: Maximum sequence length.
        padding_side: Padding side.
        truncation_side: Truncation side.

    Returns:
        Configured tokenizer.
    """
    config = get_model_config()
    tokenizer_name = tokenizer_name or config.tokenizer_name

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )

    # Configure
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    tokenizer.model_max_length = max_seq_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    return tokenizer


def get_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a valid-Jinja ChatML fallback template.

    Used when the loaded tokenizer ships no ``chat_template`` — which is the
    normal case for BASE models like unsloth/Qwen3-1.7B-Base, i.e. the live
    path during SFT data formatting for this project.

    Format matches the Qwen3 family: <|im_start|>role\ncontent<|im_end|>.
    Thinking-mode markup is intentionally omitted: v0 SFT trains direct
    (non-thinking) responses only.
    """
    template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )
    return template


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    tokenize: bool = True,
    add_generation_prompt: bool = True,
) -> str | list[int]:
    """Apply chat template to messages.

    Installs the fallback template on tokenizers that lack one, then defers
    to the standard transformers implementation.

    Args:
        tokenizer: Tokenizer instance.
        messages: List of message dicts with 'role' and 'content'.
        tokenize: Whether to return token ids (True) or a string (False).
        add_generation_prompt: Append the assistant header for generation.

    Returns:
        Formatted string or token IDs.
    """
    if tokenizer.chat_template is None:  # type: ignore[union-attr]
        tokenizer.chat_template = get_chat_template(tokenizer)  # type: ignore[union-attr]

    rendered: str | list[int] = tokenizer.apply_chat_template(  # type: ignore[union-attr,assignment]
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )
    return rendered


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str | None = None,
) -> list[dict]:
    """Build a message list for SFT.

    Args:
        instruction: Instruction text.
        input_text: Optional input text.
        output: Expected response. None builds the prompt-only form used
            for computing prompt-prefix lengths during loss masking.

    Returns:
        List of message dicts.
    """
    messages = []

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    messages.append({"role": "user", "content": user_content})

    if output:
        messages.append({"role": "assistant", "content": output})

    return messages


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))
