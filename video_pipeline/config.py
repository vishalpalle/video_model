"""Configuration for the local video understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for all pipeline stages."""

    fps: float = 1.0
    yolo_model_name: str = "yolov8n.pt"
    caption_model_name: str = "Salesforce/blip2-flan-t5-base"
    ollama_model_name: str = "mistral"
    output_dir: Path = Path("video_pipeline/outputs")
    frames_dir: Path = Path("video_pipeline/frames")
    caption_batch_size: int = 2
    ollama_url: str = "http://localhost:11434/api/generate"

    @property
    def device(self) -> str:
        """Return best available device, preferring CUDA."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def caption_torch_dtype(self) -> torch.dtype:
        """Use FP16 on GPU to optimize memory and throughput."""
        return torch.float16 if self.device == "cuda" else torch.float32
