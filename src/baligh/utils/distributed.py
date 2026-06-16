"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist
from baligh.config import get_config
from baligh.utils.logging import get_logger

logger = get_logger(__name__)


def is_distributed() -> bool:
    """Check if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get world size for distributed training."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank for multi-GPU per node."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize distributed training.
    
    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
    """
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return
    
    if dist.is_initialized():
        logger.info("Distributed already initialized")
        return
    
    # Get config
    config = get_config()
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=config.world_size,
        rank=config.local_rank,
    )
    
    # Set device
    torch.cuda.set_device(get_local_rank())
    
    logger.info(
        f"Distributed initialized: rank={get_rank()}, "
        f"world_size={get_world_size()}, local_rank={get_local_rank()}"
    )


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()
        logger.info("Distributed cleaned up")


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary values across all processes.
    
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average or sum
        
    Returns:
        Reduced dictionary
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        return {k: v for k, v in zip(names, values)}


def gather_object(obj: object) -> list:
    """Gather object from all processes.
    
    Args:
        obj: Object to gather
        
    Returns:
        List of objects from all processes (only on main process)
    """
    if not is_distributed():
        return [obj]
    
    output = [None] * get_world_size()
    dist.all_gather_object(output, obj)
    return output


def broadcast_object(obj: object, src: int = 0) -> object:
    """Broadcast object from source to all processes.
    
    Args:
        obj: Object to broadcast (only used on src)
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not is_distributed():
        return obj
    
    output = [obj]
    dist.broadcast_object_list(output, src=src)
    return output[0]
