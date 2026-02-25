"""Object detection stage using YOLOv8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm
from ultralytics import YOLO

from .utils import setup_logging


class YoloDetector:
    """Wrapper around ultralytics YOLO for frame-wise detections."""

    def __init__(self, model_name: str, device: str) -> None:
        self.logger = setup_logging()
        self.device = device
        try:
            self.model = YOLO(model_name)
            self.logger.info("Loaded YOLO model: %s", model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {exc}") from exc

    def detect(self, frame_paths: list[Path]) -> list[dict[str, Any]]:
        """Run detection on saved frames and return structured output."""
        detections: list[dict[str, Any]] = []

        for frame_path in tqdm(frame_paths, desc="YOLO detection", unit="frame"):
            try:
                results = self.model.predict(
                    source=str(frame_path),
                    device=self.device,
                    verbose=False,
                )
            except Exception as exc:
                self.logger.error("Detection failed for %s: %s", frame_path, exc)
                detections.append({"frame": frame_path.name, "objects": []})
                continue

            frame_objects: list[dict[str, Any]] = []
            result = results[0]
            names = result.names

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()
                    frame_objects.append(
                        {
                            "class": names.get(cls_id, str(cls_id)),
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 2) for v in bbox],
                        }
                    )

            detections.append({"frame": frame_path.name, "objects": frame_objects})
        return detections
