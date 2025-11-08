# preprocessing/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "Preprocessing",
    level: str = "INFO",
    log_format: str = None,
    log_file: str = None,
    console: bool = True  # <-- NEW: control console output
    ) -> logging.Logger:
    
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    #Console handler (optional)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper()))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding='utf-8')  # ← important: UTF-8 for emojis/special chars
        fh.setLevel(getattr(logging, level.upper()))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger