"""Per-stream sliding window state storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StreamState:
    """Windowed state for one stream."""

    frame_ids: deque[int]
    detections: deque[dict[str, Any]]
    ocr_history: deque[list[dict[str, Any]]]
    last_telemetry_text: set[str] = field(default_factory=set)
    last_reasoning_ts: float = 0.0
    last_object_classes: set[str] = field(default_factory=set)
    last_tracks: dict[int, tuple[float, float]] = field(default_factory=dict)


class StateManager:
    """Own all stream states and expose mutation methods."""

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self._states: dict[str, StreamState] = {}

    def get_state(self, stream_id: str) -> StreamState:
        """Get or create stream state."""
        if stream_id not in self._states:
            self._states[stream_id] = StreamState(
                frame_ids=deque(maxlen=self.window_size),
                detections=deque(maxlen=self.window_size),
                ocr_history=deque(maxlen=self.window_size),
            )
        return self._states[stream_id]

    def update(self, stream_id: str, frame_id: int, detection_packet: dict[str, Any], ocr_result: list[dict[str, Any]]) -> None:
        """Push latest frame, detections, and OCR into sliding window."""
        state = self.get_state(stream_id)
        state.frame_ids.append(frame_id)
        state.detections.append(detection_packet)
        state.ocr_history.append(ocr_result)
