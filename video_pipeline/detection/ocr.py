"""Telemetry OCR using PaddleOCR."""

from __future__ import annotations

from typing import Any

import numpy as np
from paddleocr import PaddleOCR

from video_pipeline.utils.logger import get_logger


class TelemetryOCR:
    """Extract telemetry overlays from frames with confidence filtering."""

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self.logger = get_logger()
        self.confidence_threshold = confidence_threshold
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc

    def extract(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return OCR spans as bbox/text/confidence dictionaries."""
        results = self.ocr.ocr(frame, cls=True)
        extracted: list[dict[str, Any]] = []

        for line_group in results or []:
            for item in line_group or []:
                bbox, text_meta = item[0], item[1]
                text, conf = text_meta[0], float(text_meta[1])
                if conf < self.confidence_threshold:
                    continue
                extracted.append(
                    {
                        "bbox": [[round(float(x), 2), round(float(y), 2)] for x, y in bbox],
                        "text": text,
                        "confidence": round(conf, 4),
                    }
                )
        return extracted
