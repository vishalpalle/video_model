"""CLI entrypoint for multi-stream, event-driven video understanding."""

from __future__ import annotations

import argparse
import asyncio

from config import PipelineConfig
from core.stream_manager import StreamManager
from utils.device import configure_torch_backends
from utils.logger import get_logger


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


if __name__ == "__main__":
    main()
