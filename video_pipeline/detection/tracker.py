"""Per-stream ByteTrack wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv


class StreamTrackerRegistry:
    """Maintain one ByteTrack instance per stream."""

    def __init__(self) -> None:
        self._trackers: dict[str, sv.ByteTrack] = {}

    def _tracker_for(self, stream_id: str) -> sv.ByteTrack:
        if stream_id not in self._trackers:
            self._trackers[stream_id] = sv.ByteTrack()
        return self._trackers[stream_id]

    def update(self, detection_packet: dict[str, Any]) -> dict[str, Any]:
        """Attach tracking IDs to all objects in a detection packet."""
        stream_id = detection_packet["stream_id"]
        objects = detection_packet.get("objects", [])
        if not objects:
            return detection_packet

        dets = sv.Detections(
            xyxy=np.array([o["bbox"] for o in objects], dtype=np.float32),
            confidence=np.array([o["confidence"] for o in objects], dtype=np.float32),
            class_id=np.arange(len(objects), dtype=np.int32),
        )

        tracker = self._tracker_for(stream_id)
        tracked = tracker.update_with_detections(dets)

        index_to_track_id: dict[int, int] = {}
        if tracked.tracker_id is not None:
            for cls_idx, t_id in zip(tracked.class_id.tolist(), tracked.tracker_id.tolist()):
                index_to_track_id[int(cls_idx)] = int(t_id)

        updated = []
        for idx, obj in enumerate(objects):
            item = dict(obj)
            item["tracking_id"] = index_to_track_id.get(idx)
            updated.append(item)

        out = dict(detection_packet)
        out["objects"] = updated
        return out
