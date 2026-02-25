"""Configuration objects for multi-stream video understanding."""

from __future__ import annotations

from dataclasses import dataclass, field
"""Configuration for the local video understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class PipelineConfig:
    """Top-level runtime configuration."""

    fps: float = 2.0
    sliding_window_size: int = 30
    yolo_model_name: str = "yolov8x.pt"
    qwen_model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_reasoning_tokens: int = 512
    event_threshold_seconds: float = 15.0
    batch_size: int = 8
    telemetry_confidence_threshold: float = 0.55
    output_dir: Path = Path("video_pipeline/outputs")
    frames_dir: Path = Path("video_pipeline/frames")
    per_stream_fps: dict[str, float] = field(default_factory=dict)

    @property
    def device(self) -> str:
        """Resolve the best available compute device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Use FP16 on CUDA and FP32 otherwise."""
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
