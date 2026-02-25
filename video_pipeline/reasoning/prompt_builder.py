"""Prompt construction for Qwen reasoning."""

from __future__ import annotations

import json
from typing import Any


def build_reasoning_prompt(stream_id: str, detections: list[dict[str, Any]], ocr_history: list[list[dict[str, Any]]], reasons: list[str]) -> str:
    """Create compact structured context for reasoning engine."""
    compact_timeline = []
    object_counts: dict[str, int] = {}

    for item in detections:
        frame_id = item["frame_id"]
        objects = item.get("objects", [])
        compact_timeline.append(
            {
                "frame_id": frame_id,
                "tracks": [
                    {
                        "id": obj.get("tracking_id"),
                        "class": obj.get("class"),
                        "bbox": obj.get("bbox"),
                    }
                    for obj in objects
                ],
            }
        )
        for obj in objects:
            cls = str(obj.get("class", "unknown"))
            object_counts[cls] = object_counts.get(cls, 0) + 1

    ocr_changes = [
        {
            "frame_index": idx,
            "telemetry": [entry.get("text", "") for entry in ocr_entries],
        }
        for idx, ocr_entries in enumerate(ocr_history)
    ]

    payload: dict[str, Any] = {
        "stream_id": stream_id,
        "trigger_reasons": reasons,
        "object_tracking_timeline": compact_timeline,
        "object_frequency": object_counts,
        "telemetry_changes": ocr_changes,
    }

    return (
        "You are a local video reasoning engine. Analyze the stream context and return STRICT JSON only. "
        "Required schema: {summary:string, intent:string, key_events:string[], timeline:object[], "
        "risk_assessment:string, object_behavior_analysis:string[], detected_objects_summary:object, "
        "extracted_telemetry:string[]}.\n"
        f"CONTEXT:\n{json.dumps(payload, ensure_ascii=False)}"
    )
