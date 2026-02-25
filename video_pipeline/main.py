
"""CLI entrypoint for multi-stream, event-driven video understanding."""

"""CLI entry point for end-to-end local video understanding."""

from __future__ import annotations

import argparse
import asyncio

from video_pipeline.config import PipelineConfig
from video_pipeline.core.stream_manager import StreamManager
from video_pipeline.utils.device import configure_torch_backends
from video_pipeline.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line args."""
    parser = argparse.ArgumentParser(description="Multi-stream local video understanding pipeline")
    parser.add_argument("--videos", nargs="+", required=True, help="Video file paths or RTSP URLs")
    parser.add_argument("--fps", type=float, default=None, help="Default sampling FPS for all streams")
    parser.add_argument("--batch-size", type=int, default=None, help="YOLO batch size target")
    return parser.parse_args()


async def run() -> None:
    """Run asynchronous pipeline."""
    args = parse_args()
    logger = get_logger()

    config = PipelineConfig()
    if args.fps is not None:
        config.fps = args.fps
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    configure_torch_backends()
    logger.info("Starting multi-stream pipeline with %d streams", len(args.videos))

    manager = StreamManager(config=config, videos=args.videos)
    outputs = await manager.run()
    logger.info("Pipeline complete. Streams analyzed: %d", len(outputs))
    print(outputs)


def main() -> None:
    """Synchronous bootstrap."""
    asyncio.run(run())
from pathlib import Path
from typing import Any

from config import PipelineConfig
from pipeline.aggregator import Aggregator
from pipeline.captioner import BlipCaptioner
from pipeline.detector import YoloDetector
from pipeline.sampler import sample_frames
from pipeline.tracker import ByteTrackerStage
from pipeline.utils import clear_torch_cache, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Local video understanding pipeline with YOLOv8 + ByteTrack + BLIP-2 + Ollama Mistral"
    )
    parser.add_argument("--video", required=True, type=Path, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=None, help="Sampling FPS override")
    return parser.parse_args()


def run_pipeline(video_path: Path, config: PipelineConfig) -> dict[str, Any]:
    """Execute full pipeline and return structured outputs."""
    logger = setup_logging()
    logger.info("Starting pipeline on %s", video_path)

    sampled_frames = sample_frames(video_path=video_path, output_dir=config.frames_dir, target_fps=config.fps)

    detector = YoloDetector(model_name=config.yolo_model_name, device=config.device)
    detections = detector.detect(sampled_frames)
    clear_torch_cache()

    tracker = ByteTrackerStage()
    tracked_detections = tracker.track(detections)

    captioner = BlipCaptioner(
        model_name=config.caption_model_name,
        device=config.device,
        torch_dtype=config.caption_torch_dtype,
        batch_size=config.caption_batch_size,
    )
    captions = captioner.caption(sampled_frames)
    clear_torch_cache()

    aggregator = Aggregator(ollama_url=config.ollama_url, model_name=config.ollama_model_name)
    llm_analysis = aggregator.summarize(detections=tracked_detections, captions=captions)

    final_output: dict[str, Any] = {
        "video": str(video_path),
        "config": {
            "fps": config.fps,
            "device": config.device,
            "yolo_model": config.yolo_model_name,
            "caption_model": config.caption_model_name,
            "ollama_model": config.ollama_model_name,
        },
        "captions": captions,
        "detections": tracked_detections,
        "analysis": llm_analysis,
    }

    output_path = save_json(final_output, output_dir=config.output_dir, stem=video_path.stem)
    logger.info("Saved results to %s", output_path)
    return final_output


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    config = PipelineConfig()

    if args.fps is not None:
        config.fps = args.fps

    result = run_pipeline(video_path=args.video, config=config)
    print(result)


if __name__ == "__main__":
    main()
