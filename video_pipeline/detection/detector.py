"""Batched YOLOv8x detector across streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from utils.logger import get_logger


@dataclass(slots=True)
class FramePacket:
    """Frame payload used for central batched inference."""

    stream_id: str
    frame_id: int
    frame: np.ndarray


class MultiStreamDetector:
    """Run YOLOv8x inference in batch for multiple streams."""

    def __init__(self, model_name: str, device: str, use_fp16: bool = True) -> None:
        self.logger = get_logger()
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        try:
            self.model = YOLO(model_name)
            self.logger.info("Loaded detector model %s on %s", model_name, device)
        except Exception as exc:
            raise RuntimeError(f"Unable to load YOLO model '{model_name}': {exc}") from exc

    def detect_batch(self, packets: list[FramePacket]) -> list[dict[str, Any]]:
        """Run a single batched inference call and normalize result format."""
        if not packets:
            return []

        frames = [p.frame for p in packets]
        results = self.model.predict(
            source=frames,
            device=self.device,
            half=self.use_fp16,
            verbose=False,
        )

        payload: list[dict[str, Any]] = []
        for packet, result in zip(packets, results):
            objects: list[dict[str, Any]] = []
            names = result.names
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    objects.append(
                        {
                            "class": names.get(cls_id, str(cls_id)),
                            "confidence": round(float(box.conf.item()), 4),
                            "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                            "tracking_id": None,
                        }
                    )

            payload.append(
                {
                    "stream_id": packet.stream_id,
                    "frame_id": packet.frame_id,
                    "objects": objects,
                }
            )

        return payload
