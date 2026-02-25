"""Event rules for triggering reasoning."""

from __future__ import annotations

import math
import time
from typing import Any

from video_pipeline.core.state_manager import StreamState


class EventManager:
    """Evaluate stream changes and decide if reasoning should run."""

    def __init__(self, event_threshold_seconds: float, speed_change_threshold: float = 25.0) -> None:
        self.event_threshold_seconds = event_threshold_seconds
        self.speed_change_threshold = speed_change_threshold

    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def should_trigger(self, state: StreamState, detection_packet: dict[str, Any], ocr_result: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        """Trigger on object changes, speed jumps, telemetry changes, or timeout."""
        reasons: list[str] = []

        current_classes = {obj["class"] for obj in detection_packet.get("objects", [])}
        new_classes = current_classes - state.last_object_classes
        disappeared = state.last_object_classes - current_classes
        if new_classes:
            reasons.append(f"new_object_class:{sorted(new_classes)}")
        if disappeared:
            reasons.append(f"object_disappeared:{sorted(disappeared)}")

        track_positions: dict[int, tuple[float, float]] = {}
        for obj in detection_packet.get("objects", []):
            tid = obj.get("tracking_id")
            if tid is None:
                continue
            center = self._bbox_center(obj["bbox"])
            track_positions[int(tid)] = center
            if tid in state.last_tracks:
                prev = state.last_tracks[tid]
                dist = math.dist(prev, center)
                if dist > self.speed_change_threshold:
                    reasons.append(f"speed_change_track:{tid}")

        current_telemetry = {x["text"] for x in ocr_result}
        if current_telemetry != state.last_telemetry_text:
            reasons.append("telemetry_change")

        now = time.time()
        if now - state.last_reasoning_ts >= self.event_threshold_seconds:
            reasons.append("interval_elapsed")

        state.last_object_classes = current_classes
        state.last_tracks = track_positions
        state.last_telemetry_text = current_telemetry

        return (len(reasons) > 0, reasons)
