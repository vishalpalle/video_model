"""Shared helpers for logging, IO, and cleanup."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


LOGGER_NAME = "video_pipeline"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-wide logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], output_dir: Path, stem: str) -> Path:
    """Persist JSON data with a timestamped filename."""
    ensure_dir(output_dir)
    output_path = output_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def clear_torch_cache() -> None:
    """Release cached GPU memory where available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
