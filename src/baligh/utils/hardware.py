"""Hardware capability detection for precision and attention selection.

Resolves mixed-precision and attention-implementation settings against the
actual GPU compute capability, so the same defaults work on Turing (T4,
Quadro RTX 5000, sm_75) and Ampere+ (A100, L4, sm_80+).

Rules:
- bf16 tensor cores require sm_80+; on older GPUs fall back to fp16.
- FlashAttention-2 requires sm_80+; on older GPUs fall back to "sdpa".
- On CPU-only systems precision falls back to "no" and attention to "sdpa".
"""

import torch

from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def get_device_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability of the current CUDA device.

    Returns None when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:  # pragma: no cover - defensive
        return None


def resolve_precision(requested: str = "auto") -> str:
    """Resolve a requested mixed-precision mode to one supported by the hardware.

    Args:
        requested: "auto", "bf16", "fp16", or "no".

    Returns:
        One of "bf16", "fp16", "no".
    """
    capability = get_device_capability()

    if requested == "no":
        return "no"
    if requested == "fp16":
        return "fp16"
    if requested in ("bf16", "auto"):
        if capability is None:
            return "no" if requested == "auto" else "fp16"
        if capability >= (8, 0):
            return "bf16"
        logger.info(
            "GPU compute capability %s.%s does not support bf16; falling back to fp16",
            capability[0],
            capability[1],
        )
        return "fp16"
    raise ValueError(f"Unknown precision mode: {requested}")


def resolve_attn_implementation(requested: str | None = None) -> str:
    """Resolve the attention implementation supported by the hardware.

    Args:
        requested: Preferred implementation ("flash_attention_2", "sdpa",
            or None/"" for auto-detection).

    Returns:
        "flash_attention_2" only on sm_80+ GPUs when requested; otherwise
        "sdpa".
    """
    if not requested:
        return "sdpa"

    capability = get_device_capability()
    if requested == "flash_attention_2":
        if capability is not None and capability >= (8, 0):
            return "flash_attention_2"
        logger.info(
            "FlashAttention-2 requires Ampere+ (sm_80+); using sdpa instead "
            f"(device capability: {capability})"
        )
        return "sdpa"
    return requested


def resolve_torch_dtype(precision: str | None = None) -> torch.dtype:
    """Map a resolved precision mode to a torch dtype for weight loading."""
    if precision is None:
        precision = resolve_precision("auto")
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }
    return mapping.get(precision, torch.float16)
