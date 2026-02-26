"""FastAPI app exposing the video pipeline with Swagger UI."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from video_pipeline.config import PipelineConfig
from video_pipeline.core.stream_manager import StreamManager


class AnalyzeRequest(BaseModel):
    """Request payload for analyzing one or more video streams."""

    videos: list[str] = Field(..., min_length=1, description="Video file paths or RTSP URLs")
    fps: float | None = Field(default=None, gt=0, description="Sampling FPS override")
    batch_size: int | None = Field(default=None, gt=0, description="Batch size override")


class AnalyzeResponse(BaseModel):
    """Pipeline response payload."""

    outputs: dict[str, dict]


app = FastAPI(
    title="Video Pipeline API",
    description="Run the multi-stream video understanding pipeline via REST endpoints.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health endpoint used for smoke testing."""

    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run pipeline analysis and return structured stream outputs."""

    config = PipelineConfig()
    if request.fps is not None:
        config.fps = request.fps
    if request.batch_size is not None:
        config.batch_size = request.batch_size

    try:
        manager = StreamManager(config=config, videos=request.videos)
        outputs = await manager.run()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalyzeResponse(outputs=outputs)
