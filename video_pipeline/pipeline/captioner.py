"""Caption generation stage using BLIP-2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from .utils import clear_torch_cache, setup_logging


class BlipCaptioner:
    """Generate captions for sampled frames using BLIP-2."""

    def __init__(self, model_name: str, device: str, torch_dtype: torch.dtype, batch_size: int = 2) -> None:
        self.logger = setup_logging()
        self.device = device
        self.batch_size = batch_size

        try:
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            self.model.to(device)
            self.model.eval()
            self.logger.info("Loaded BLIP-2 model: %s", model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load caption model '{model_name}': {exc}") from exc

    def caption(self, frame_paths: list[Path]) -> list[dict[str, Any]]:
        """Generate one caption per frame, with optional batching."""
        captions: list[dict[str, Any]] = []

        for start in tqdm(range(0, len(frame_paths), self.batch_size), desc="Captioning", unit="batch"):
            batch_paths = frame_paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]

            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=40)
                texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for path, text in zip(batch_paths, texts):
                captions.append({"frame": path.name, "caption": text.strip()})

            for img in images:
                img.close()

        clear_torch_cache()
        return captions
