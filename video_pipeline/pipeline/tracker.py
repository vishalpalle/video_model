"""Tracking stage using ByteTrack from supervision."""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv
from tqdm import tqdm

from .utils import setup_logging


class ByteTrackerStage:
    """Assign persistent tracking IDs to detections."""

    def __init__(self) -> None:
        self.logger = setup_logging()
        self.tracker = sv.ByteTrack()

    def track(self, detections_by_frame: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach tracking IDs to each detection in sequence."""
        tracked_output: list[dict[str, Any]] = []

        for item in tqdm(detections_by_frame, desc="ByteTrack", unit="frame"):
            objects = item.get("objects", [])
            if not objects:
                tracked_output.append(item)
                continue

            xyxy = np.array([obj["bbox"] for obj in objects], dtype=np.float32)
            confidence = np.array([obj["confidence"] for obj in objects], dtype=np.float32)
            class_id = np.array([idx for idx, _ in enumerate(objects)], dtype=np.int32)

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

            tracked_detections = self.tracker.update_with_detections(sv_detections)

            id_map: dict[int, int] = {}
            if tracked_detections.tracker_id is not None:
                for det_idx, track_id in zip(tracked_detections.class_id.tolist(), tracked_detections.tracker_id.tolist()):
                    id_map[int(det_idx)] = int(track_id)

            frame_objects: list[dict[str, Any]] = []
            for idx, obj in enumerate(objects):
                updated = dict(obj)
                updated["tracking_id"] = id_map.get(idx)
                frame_objects.append(updated)

            tracked_output.append({"frame": item["frame"], "objects": frame_objects})

        self.logger.info("Tracking complete for %d frames.", len(tracked_output))
        return tracked_output
