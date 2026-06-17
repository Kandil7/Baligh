"""Modal inference server for Baligh-1.5B.

Usage:
    # Deploy as web endpoint
    modal deploy src/modal/serve.py

    # Test locally
    modal serve src/modal/serve.py

    # Call from Python
    modal run src/modal/infer.py --prompt "ما هي عاصمة مصر؟"
"""

import modal
from src.modal.app import app, vol, VOL_PATH


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    allow_concurrent_inputs=10,
)
@modal.web_endpoint(method="GET")
def generate(
    prompt: str = "ما هي عاصمة مصر؟",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate text from Baligh model via web endpoint."""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Load model
    from baligh.inference import TextGenerator

    model_path = f"{VOL_PATH}/training/sft/final"
    if not Path(model_path).exists():
        model_path = "Kandil7/Baligh-1.5B"  # Fallback to HF Hub

    generator = TextGenerator(model_path)
    response = generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return {
        "prompt": prompt,
        "response": response,
        "model": model_path,
    }


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token")],
    allow_concurrent_inputs=10,
)
@modal.web_endpoint(method="GET")
def chat(
    message: str = "مرحبا، كيف حالك؟",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Multi-turn chat via web endpoint."""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from baligh.inference import ChatBot

    model_path = f"{VOL_PATH}/training/sft/final"
    if not Path(model_path).exists():
        model_path = "Kandil7/Baligh-1.5B"

    chatbot = ChatBot(model_path)
    response = chatbot.chat(message, max_new_tokens=max_new_tokens, temperature=temperature)

    return {
        "message": message,
        "response": response,
        "history": chatbot.history,
    }


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def infer_cli(
    prompt: str = "ما هي عاصمة مصر؟",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """CLI inference: modal run src/modal/infer.py --prompt '...'"""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from baligh.inference import TextGenerator

    model_path = f"{VOL_PATH}/training/sft/final"
    if not Path(model_path).exists():
        model_path = "Kandil7/Baligh-1.5B"

    print(f"Loading model: {model_path}")
    generator = TextGenerator(model_path)
    response = generator.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print(f"Response: {response}")
    print(f"{'='*60}\n")


@app.local_entrypoint()
def main(
    prompt: str = "ما هي عاصمة مصر؟",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Local entrypoint: modal run src/modal/infer.py --prompt '...'"""
    infer_cli.remote(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
