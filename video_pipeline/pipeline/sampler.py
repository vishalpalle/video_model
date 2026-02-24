"""Frame sampling utilities using OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

from .utils import ensure_dir, setup_logging


def sample_frames(video_path: Path, output_dir: Path, target_fps: float) -> list[Path]:
    """Sample frames from a video at a configurable FPS and save them to disk."""
    logger = setup_logging()
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        logger.warning("Invalid source FPS detected; defaulting to 30 FPS.")
        source_fps = 30.0

    frame_interval = max(int(round(source_fps / target_fps)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_frames: list[Path] = []
    frame_idx = 0
    sampled_idx = 0

    logger.info(
        "Sampling frames from %s at %.2f FPS (source FPS %.2f, interval=%d)",
        video_path,
        target_fps,
        source_fps,
        frame_interval,
    )

    with tqdm(total=total_frames, desc="Sampling frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = output_dir / f"frame_{sampled_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(frame_path)
                sampled_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    logger.info("Sampled %d frames.", len(saved_frames))
    return saved_frames
