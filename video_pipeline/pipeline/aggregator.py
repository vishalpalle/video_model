"""Aggregate signals and query local Ollama/Mistral for video understanding."""

from __future__ import annotations

import json
from typing import Any

import requests

from .utils import setup_logging


class Aggregator:
    """Build context prompt from detections/captions and call local LLM."""

    def __init__(self, ollama_url: str, model_name: str) -> None:
        self.logger = setup_logging()
        self.ollama_url = ollama_url
        self.model_name = model_name

    def _build_prompt(self, detections: list[dict[str, Any]], captions: list[dict[str, Any]]) -> str:
        data = {"captions": captions, "detections": detections}
        return (
            "You are a video understanding assistant. Analyze the following per-frame outputs "
            "from object detection, tracking, and captioning. "
            "Return STRICT JSON with keys: summary (string), intent (string), key_events (array of strings), "
            "timeline (array of objects with frame and event).\n\n"
            f"DATA:\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    def _extract_response_text(self, response_json: dict[str, Any]) -> str:
        if "response" in response_json:
            return str(response_json["response"])

        if "message" in response_json and isinstance(response_json["message"], dict):
            return str(response_json["message"].get("content", ""))

        return json.dumps(response_json)

    def summarize(self, detections: list[dict[str, Any]], captions: list[dict[str, Any]]) -> dict[str, Any]:
        """Query local Ollama model and return structured JSON when available."""
        prompt = self._build_prompt(detections, captions)
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=300)
            response.raise_for_status()
        except requests.RequestException as exc:
            error_message = (
                "Could not reach local Ollama server. Ensure Ollama is running and model is available. "
                f"Details: {exc}"
            )
            self.logger.error(error_message)
            return {"error": error_message}

        text = self._extract_response_text(response.json()).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            self.logger.warning("LLM response was not valid JSON; returning raw text.")

        return {"raw_text": text}
