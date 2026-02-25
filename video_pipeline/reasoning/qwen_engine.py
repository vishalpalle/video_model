"""Qwen2-VL local reasoning engine."""

from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from video_pipeline.utils.device import clear_device_cache
from video_pipeline.utils.logger import get_logger


class QwenReasoningEngine:
    """Run local structured reasoning with Qwen2-VL."""

    def __init__(self, model_path: str, torch_dtype: torch.dtype, max_new_tokens: int) -> None:
        self.logger = get_logger()
        self.max_new_tokens = max_new_tokens
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self.logger.info("Loaded Qwen reasoning model: %s", model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Qwen model '{model_path}': {exc}") from exc

    def _run_once(self, prompt: str) -> str:
        inputs = self.processor(text=[prompt], return_tensors="pt")
        input_ids = inputs.get("input_ids")
        prompt_length = int(input_ids.shape[-1]) if input_ids is not None else 0

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_tokens = out[:, prompt_length:] if prompt_length > 0 else out
        text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text

    def infer(self, prompt: str) -> dict[str, Any]:
        """Run inference and enforce JSON parsing with one retry."""
        errors: list[str] = []
        for attempt in range(2):
            response_text = self._run_once(prompt)
            candidate = response_text[response_text.find("{") : response_text.rfind("}") + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    clear_device_cache()
                    return parsed
            except json.JSONDecodeError as exc:
                errors.append(f"attempt_{attempt + 1}:{exc}")
                self.logger.warning("Reasoning parse failed on attempt %d: %s", attempt + 1, exc)

        clear_device_cache()
        return {"raw_text": response_text, "error": "JSON parsing failed", "attempt_errors": errors}
