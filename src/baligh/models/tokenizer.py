"""Tokenizer utilities for Baligh-1.5B v0."""

from transformers import AutoTokenizer
from baligh.config import get_model_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def get_tokenizer(
    tokenizer_name: str | None = None,
    max_seq_length: int = 2048,
    padding_side: str = "right",
    truncation_side: str = "right",
    add_special_tokens: bool = True,
) -> AutoTokenizer:
    """Get configured tokenizer.
    
    Args:
        tokenizer_name: Tokenizer name or path.
        max_seq_length: Maximum sequence length.
        padding_side: Padding side.
        truncation_side: Truncation side.
        add_special_tokens: Whether to add special tokens.
        
    Returns:
        Configured tokenizer.
    """
    config = get_model_config()
    tokenizer_name = tokenizer_name or config.tokenizer_name
    
    tokenizer = AutoTokenizer.from_pretrained(
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


def get_chat_template(tokenizer: AutoTokenizer) -> str:
    """Get chat template for Qwen2.5.
    
    Args:
        tokenizer: Tokenizer instance.
        
    Returns:
        Chat template string.
    """
    # Qwen2.5 chat template
    template = """{{% for message in messages %}}
{{% if message['role'] == 'system' %}}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{{% elif message['role'] == 'user' %}}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{{% elif message['role'] == 'assistant' %}}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{{% endif }}
{{% endfor %}}
{{% if add_generation_prompt %}}
<|im_start|>assistant
{{% endif %}}"""
    return template


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict],
    tokenize: bool = True,
    add_generation_prompt: bool = True,
) -> str | list[int]:
    """Apply chat template to messages.
    
    Args:
        tokenizer: Tokenizer instance.
        messages: List of message dicts with 'role' and 'content'.
        tokenize: Whether to tokenize output.
        add_generation_prompt: Whether to add generation prompt.
        
    Returns:
        Formatted string or token IDs.
    """
    if tokenizer.chat_template is None:
        tokenizer.chat_template = get_chat_template(tokenizer)
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
) -> list[dict]:
    """Format instruction for SFT.
    
    Args:
        instruction: Instruction text.
        input_text: Optional input text.
        output: Expected output (for training).
        
    Returns:
        List of message dicts.
    """
    messages = []
    
    if input_text:
        user_content = f"{instruction}

{input_text}"
    else:
        user_content = instruction
    
    messages.append({"role": "user", "content": user_content})
    
    if output:
        messages.append({"role": "assistant", "content": output})
    
    return messages


def count_tokens(tokenizer: AutoTokenizer, text: str) -> int:
    """Count tokens in text.
    
    Args:
        tokenizer: Tokenizer instance.
        text: Text to count tokens for.
        
    Returns:
        Number of tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))
