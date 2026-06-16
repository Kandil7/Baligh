"""Logging setup for Baligh-1.5B v0."""

import sys
from pathlib import Path
from loguru import logger
from baligh.config import get_config


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_file: Path | None = None,
    json_logs: bool = False,
) -> None:
    """Configure loguru logger."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    if log_format == "json":
        logger.add(
            sys.stdout,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,
            colorize=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=json_logs,
        )
    
    # Configure standard library logging to use loguru
    import logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


class InterceptHandler(logging.Handler):
    """Intercept standard library logging and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# Initialize on import
config = get_config()
setup_logging(
    log_level=config.log_level,
    log_format=config.log_format,
    log_file=config.logs_dir / "baligh.log" if config.logs_dir else None,
    json_logs=config.log_format == "json",
)
