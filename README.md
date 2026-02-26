# video_model

## Swagger API for testing

A FastAPI app is available at `video_pipeline/api.py` with built-in Swagger UI.

### Run

```bash
uvicorn video_pipeline.api:app --host 0.0.0.0 --port 8000 --reload
```

### Open Swagger UI

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

- `GET /health`: health check.
- `POST /analyze`: runs the multi-stream analysis pipeline.
