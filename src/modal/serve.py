"""Modal inference server for Baligh-1.7B.

Endpoints are POST + API-key gated. Create the secret once:
    modal secret create baligh-api-key BALIGH_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")

Then:
    modal deploy src/modal/serve.py
    curl -X POST <url> -H "x-api-key: $BALIGH_API_KEY" \
         -H "Content-Type: application/json" -d '{"prompt": "..."}'
"""

import modal
from src.modal.app import VOL_PATH, api_key_secret, app, vol

MODEL_FALLBACK = "Kandil7/Baligh-1.7B"


def _require_api_key(headers) -> None:
    """Reject unauthenticated requests before any GPU work happens."""
    import os

    expected = os.environ.get("BALIGH_API_KEY")
    provided = headers.get("x-api-key")
    if not expected:
        raise RuntimeError("Server misconfigured: BALIGH_API_KEY secret missing")
    if not provided or provided != expected:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header")


def _resolve_model_path() -> str:
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"
    model_path = f"{VOL_PATH}/training/sft/final"
    if Path(model_path).exists():
        return model_path
    return MODEL_FALLBACK


def _login_hf() -> None:
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token"), api_key_secret],
    allow_concurrent_inputs=10,
)
@modal.web_endpoint(method="POST")
def generate(request: dict):
    """Single-turn generation. Body: {"prompt", "max_new_tokens", ...}."""
    import fastapi

    _require_api_key(fastapi.request.headers)
    _login_hf()

    from baligh.inference import TextGenerator

    prompt = request.get("prompt") or "مرحبا"
    generator = TextGenerator(_resolve_model_path())
    response = generator.generate(
        prompt,
        max_new_tokens=request.get("max_new_tokens"),
        temperature=request.get("temperature"),
        top_p=request.get("top_p"),
    )
    return {"response": response, "model": _resolve_model_path()}


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token"), api_key_secret],
    allow_concurrent_inputs=10,
)
@modal.web_endpoint(method="POST")
def chat_endpoint(request: dict):
    """Stateless chat turn. Body: {"message", "history": [...]}."""
    import fastapi

    _require_api_key(fastapi.request.headers)
    _login_hf()

    from baligh.inference import ChatBot

    bot = ChatBot(_resolve_model_path())
    for turn in (request.get("history") or [])[-10:]:
        if turn.get("role") in ("user", "assistant"):
            bot.history.append({"role": turn["role"], "content": str(turn["content"])})

    message = request.get("message") or "مرحبا"
    response = bot.chat(message, max_new_tokens=request.get("max_new_tokens", 512))
    return {"response": response}


@app.function(
    gpu="a10g",
    volumes={VOL_PATH: vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def infer_cli(
    prompt: str = "ما هي عاصمة مصر؟",
    max_new_tokens: int = 512,
):
    """CLI inference: modal run src/modal/serve.py --prompt '...'"""
    import os

    os.environ["HF_HOME"] = f"{VOL_PATH}/.cache/huggingface"
    _login_hf()

    from baligh.inference import TextGenerator

    print(f"Loading model: {_resolve_model_path()}")
    generator = TextGenerator(_resolve_model_path())
    response = generator.generate(prompt, max_new_tokens=max_new_tokens)

    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)
    print(f"Response: {response}")
    print("=" * 60 + "\n")


@app.local_entrypoint()
def main(prompt: str = "ما هي عاصمة مصر؟", max_new_tokens: int = 512):
    infer_cli.remote(prompt=prompt, max_new_tokens=max_new_tokens)
