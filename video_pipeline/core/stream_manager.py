"""Async multi-stream orchestrator for detection, tracking, OCR, and reasoning."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from config import PipelineConfig
from core.event_manager import EventManager
from core.state_manager import StateManager
from detection.detector import FramePacket, MultiStreamDetector
from detection.ocr import TelemetryOCR
from detection.tracker import StreamTrackerRegistry
from reasoning.prompt_builder import build_reasoning_prompt
from reasoning.qwen_engine import QwenReasoningEngine
from utils.logger import get_logger


@dataclass(slots=True)
class StreamContext:
    """Video source runtime info."""

    stream_id: str
    source: str
    capture: cv2.VideoCapture
    frame_id: int = 0
    ended: bool = False
    next_sample_ts: float = 0.0


class StreamManager:
    """Manage all streams and process them in central batches."""

    def __init__(self, config: PipelineConfig, videos: list[str]) -> None:
        self.config = config
        self.logger = get_logger()
        self.stop_event = asyncio.Event()
        self.state = StateManager(window_size=config.sliding_window_size)
        self.events = EventManager(event_threshold_seconds=config.event_threshold_seconds)
        self.detector = MultiStreamDetector(config.yolo_model_name, config.device, use_fp16=True)
        self.trackers = StreamTrackerRegistry()
        self.ocr = TelemetryOCR(confidence_threshold=config.telemetry_confidence_threshold)
        self.reasoner = QwenReasoningEngine(
            model_path=config.qwen_model_path,
            torch_dtype=config.torch_dtype,
            max_new_tokens=config.max_reasoning_tokens,
        )
        self.streams = self._init_streams(videos)

    def _init_streams(self, videos: list[str]) -> list[StreamContext]:
        streams: list[StreamContext] = []
        for idx, src in enumerate(videos):
            stream_id = f"stream_{idx}"
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.logger.error("Unable to open stream %s (%s)", stream_id, src)
                continue
            streams.append(StreamContext(stream_id=stream_id, source=src, capture=cap))
        if not streams:
            raise RuntimeError("No valid streams available. Check video paths or RTSP URLs.")
        return streams

    def _on_signal(self, *_: Any) -> None:
        self.logger.info("Shutdown signal received, stopping gracefully...")
        self.stop_event.set()

    def _target_fps(self, stream_id: str) -> float:
        return self.config.per_stream_fps.get(stream_id, self.config.fps)

    async def _collect_sampled_frames(self) -> list[FramePacket]:
        packets: list[FramePacket] = []
        now = time.time()
        for stream in self.streams:
            if stream.ended:
                continue

            target_fps = self._target_fps(stream.stream_id)
            sample_period = 1.0 / max(target_fps, 0.1)
            if now < stream.next_sample_ts:
                continue

            ok, frame = await asyncio.to_thread(stream.capture.read)
            if not ok or frame is None:
                stream.ended = True
                self.logger.warning("Stream ended/corrupted: %s", stream.stream_id)
                continue

            stream.frame_id += 1
            stream.next_sample_ts = now + sample_period

            frame_path = self.config.frames_dir / f"{stream.stream_id}_frame_{stream.frame_id:07d}.jpg"
            await asyncio.to_thread(cv2.imwrite, str(frame_path), frame)
            packets.append(FramePacket(stream_id=stream.stream_id, frame_id=stream.frame_id, frame=frame))

        return packets

    async def _reason_about_stream(
        self,
        stream_id: str,
        reasons: list[str],
    ) -> dict[str, Any]:
        state = self.state.get_state(stream_id)
        prompt = build_reasoning_prompt(
            stream_id=stream_id,
            detections=list(state.detections),
            ocr_history=list(state.ocr_history),
            reasons=reasons,
        )
        result = await asyncio.to_thread(self.reasoner.infer, prompt)

        object_summary: dict[str, int] = {}
        for packet in state.detections:
            for obj in packet.get("objects", []):
                cls = str(obj.get("class", "unknown"))
                object_summary[cls] = object_summary.get(cls, 0) + 1

        telemetry = sorted({entry.get("text", "") for batch in state.ocr_history for entry in batch if entry.get("text")})
        result.setdefault("detected_objects_summary", object_summary)
        result.setdefault("extracted_telemetry", telemetry)
        state.last_reasoning_ts = time.time()
        return result

    async def run(self) -> dict[str, dict[str, Any]]:
        """Main processing loop for all streams."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.frames_dir.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        final_outputs: dict[str, dict[str, Any]] = {}

        with tqdm(desc="Multi-stream processing", unit="batch") as pbar:
            while not self.stop_event.is_set():
                packets = await self._collect_sampled_frames()
                if not packets and all(s.ended for s in self.streams):
                    break
                if not packets:
                    await asyncio.sleep(0.01)
                    continue

                detections = await asyncio.to_thread(self.detector.detect_batch, packets)

                for packet, det in zip(packets, detections):
                    tracked = await asyncio.to_thread(self.trackers.update, det)
                    ocr_res = await asyncio.to_thread(self.ocr.extract, packet.frame)

                    self.state.update(packet.stream_id, packet.frame_id, tracked, ocr_res)
                    stream_state = self.state.get_state(packet.stream_id)
                    triggered, reasons = self.events.should_trigger(stream_state, tracked, ocr_res)

                    if triggered:
                        reasoning_output = await self._reason_about_stream(packet.stream_id, reasons)
                        final_outputs[packet.stream_id] = {
                            "stream_id": packet.stream_id,
                            "source": next(s.source for s in self.streams if s.stream_id == packet.stream_id),
                            "last_frame_id": packet.frame_id,
                            "summary": reasoning_output.get("summary", ""),
                            "intent": reasoning_output.get("intent", ""),
                            "key_events": reasoning_output.get("key_events", []),
                            "timeline": reasoning_output.get("timeline", []),
                            "detected_objects_summary": reasoning_output.get("detected_objects_summary", {}),
                            "extracted_telemetry": reasoning_output.get("extracted_telemetry", []),
                            "risk_assessment": reasoning_output.get("risk_assessment", ""),
                            "object_behavior_analysis": reasoning_output.get("object_behavior_analysis", []),
                            "trigger_reasons": reasons,
                        }

                pbar.update(1)

        for stream in self.streams:
            stream.capture.release()

        for stream_id, payload in final_outputs.items():
            output_path = self.config.output_dir / f"{stream_id}_analysis.json"
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self.logger.info("Saved stream analysis: %s", output_path)

        return final_outputs
