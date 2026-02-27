### High-level goals

To make `src/server/main.py` production ready, you want to:

- **Stabilize configuration and secrets**
- **Harden security and validation**
- **Improve reliability and performance (I/O, external APIs, temp files)**
- **Clarify API design and structure**
- **Add observability, tests, and a robust deployment setup**

Below is a step‑by‑step plan tied to concrete issues visible in `main.py`.

---

### 1. Restructure the server module

- **Split `main.py` into logical modules**
  - Create separate modules/routers for concerns:
    - `chat.py` (chat, indic_chat, chat_direct)
    - `audio.py` (TTS, ASR, transcribe, speech-to-speech)
    - `pdf.py` (all PDF endpoints)
    - `vision.py` / `ocr.py` (OCR, visual query)
    - `health.py` / `utility.py` (health, root redirect)
  - Keep only:
    - `FastAPI` app creation
    - global configuration (settings, logging)
    - router inclusion
    - `__main__` / ASGI entrypoint in `main.py`.

- **Centralize shared helpers**
  - Move things like `get_openai_client`, `validate_model`, `validate_language`, `time_to_words`, `encode_image`, `ocr_page_with_rolm_query`, PDF extraction helpers into `services/` or `utils/` modules.
  - Remove clearly duplicated Pydantic classes (`VisualQueryRequest`, `VisualQueryResponse` appear twice with slightly different shapes, multiple `BaseModel` imports, repeated `SUPPORTED_LANGUAGES` checks, etc.) and keep one canonical definition per concept.

- **Use FastAPI routers**
  - Create routers (`APIRouter`) per domain, with tags:
    - e.g. `chat_router = APIRouter(prefix="/v1", tags=["Chat"])`
  - Include them in `app` in one place:
    - `app.include_router(chat_router)`, etc.
  - This improves readability and makes it easier to add middlewares or dependencies per group.

---

### 2. Configuration, secrets, and environment management

- **Replace hardcoded values with proper settings**
  - `Settings.openai_api_key = "http"` and `get_openai_client` using `OpenAI(api_key="http", base_url=...)` are not production suitable.
  - Define a `Settings` class using `pydantic_settings.BaseSettings`:
    - API keys (LLM, TTS, ASR, translation)
    - External base URLs (`DWANI_API_BASE_URL_LLM`, `_TTS`, `_ASR`, `_TRANSLATE`, `_S2S`, `_PDF`)
    - Rate-limit configuration, logging level, max payload sizes, etc.
  - Load from environment and `.env` files; never hardcode secrets in code or Git.

- **Unify `SUPPORTED_MODELS` and language lists**
  - You have:
    - Global `SUPPORTED_MODELS = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]`
    - Per-endpoint `valid_models = ["gemma3", "qwen3", "sarvam-m", "gpt-oss"]`
    - `get_openai_client` currently only allows `["gemma3"]`.
  - Define a single source of truth, e.g.:
    - `ModelConfig` (dict with allowed models, backends, capabilities).
  - Similarly, unify `allowed_languages` / `SUPPORTED_LANGUAGES` so lists aren’t copy‑pasted across endpoints.

- **Formalize runtime environment profiles**
  - Configure dev/stage/prod via env var (`ENVIRONMENT`) and:
    - Different log levels
    - CORS origins
    - Mock vs real external services.

---

### 3. Logging and observability

- **Refine logging configuration**
  - Logging config is already using `RotatingFileHandler` and stdout, which is good, but:
    - Make filename/path configurable (`LOG_DIR`, `LOG_FILE`).
    - Prefer JSON logs in production for easier ingestion (ELK, Loki, etc.).
  - Use structured logging fields consistently:
    - Request id, user id (if available), endpoint, external URL, durations, error codes.

- **Add request/response metrics and tracing**
  - Integrate:
    - Metrics (Prometheus via `prometheus_fastapi_instrumentator` or similar).
    - Basic tracing (OpenTelemetry if you have the infra).
  - Record external call latency for:
    - LLM, TTS, ASR, translation, PDF services.

- **Improve health and readiness endpoints**
  - `/v1/health` now returns a static `"model": "llm_model_name"`.
  - Extend to:
    - Liveness: app is running.
    - Readiness: quickly check connectivity to required dependencies (e.g. try a cheap request or head to each critical external service with a timeout guard, and/or track via startup checks).

---

### 4. Security hardening

- **Tighten CORS**
  - CORS currently allows patterns like `"https://*.dwani.ai"`, `"https://*.hf.space"`, `http://localhost:11080`.
  - For production:
    - Use explicit allowed origins (prod UI domains).
    - Put dev origins behind environment flag.
    - Consider removing `*`-like patterns and keeping them minimal.

- **Add authentication/authorization**
  - Right now, all endpoints are open:
    - Introduce API key or JWT-based auth:
      - Define dependency `get_current_client` / `verify_api_key`.
      - Apply globally or per-router.
  - Segment capabilities:
    - Some heavy/expensive endpoints (PDF processing, LLM vision) might be limited to privileged clients.

- **Harden file and input handling**
  - For file uploads (`PDF`, `image/png`, audio):
    - Enforce `max_content_length` at reverse proxy (nginx) and app level.
    - Validate MIME types and file extensions; currently you mostly check `filename.lower().endswith(".pdf")` or `content_type.startswith("image/png")` but not size.
    - Use a dedicated temp directory, not default system temp, and configure cleanup.
  - For text inputs:
    - You have length checks on prompts and TTS text – keep and centralize these as constants to avoid inconsistencies.

- **Protect secrets and external URLs**
  - Ensure:
    - No secrets logged.
    - Base URLs validated or loaded from trusted config only (avoid any user influence).
  - For OpenAI‑compatible APIs:
    - Use an API key from environment.
    - Consider using organization/tenant isolation on backend.

---

### 5. External services and reliability

- **Standardize HTTP client usage**
  - Many endpoints use `requests.post` synchronously inside async functions:
    - Move to `httpx.AsyncClient` with a shared client:
      - Create in `@app.on_event("startup")`, close in `@app.on_event("shutdown")`.
      - Configure timeouts and connection pooling.
    - If you must use `requests`, wrap heavy calls in threadpool (`run_in_threadpool`) to avoid blocking event loop.

- **Implement retries, circuit breakers, and fallback behavior**
  - Add retry logic for transient failures (`connect`/`read` timeouts, 5xx from external services).
  - Decide which endpoints:
    - Fail fast vs retry.
    - Use simple backoff.
  - Expose partial failures clearly in responses; ensure you don’t return half‑baked results.

- **Normalize timeouts**
  - You have a mix of `timeout=30` and no timeout in some `requests.post`.
  - Define constants like `EXTERNAL_API_TIMEOUT = 15` and reuse everywhere.

---

### 6. API design correctness and cleanup

- **Fix response models vs actual responses**
  - `chat_v2` endpoint is annotated `response_model=ChatResponse` but returns `ChatDirectResponse` (field name `response`). Harmonize:
    - Use one `ChatResponse` model per style, or rename for clarity.
  - Ensure all endpoints actually conform to their `response_model` (PDF endpoints, OCR, etc.).

- **Remove dead or inconsistent endpoints**
  - `indic_visual_query_direct` is defined but not decorated with `@app.<method>` – it’s not exposed.
    - Decide whether to expose it properly or move it entirely behind another endpoint as a helper.
  - Clean commented‑out YOLO/object detection code or move to a clearly experimental module if needed.

- **Normalize naming and versioning**
  - Current paths mix styles like `/v1/indic_chat`, `/v1/chat_direct`, `/v1/transcribe/` (with trailing slash) and `/ocr`.
  - Decide on:
    - Consistent base path (`/v1` for everything).
    - Resource names (`/audio/transcribe`, `/audio/speech`, `/chat`, `/pdf/summarize`, etc.).
  - Introduce a deprecation strategy if you already have clients:
    - Mark old endpoints with `deprecated=True` in FastAPI docs and keep them until clients migrate.

---

### 7. Performance and scalability

- **Make I/O fully non‑blocking**
  - Convert all external HTTP calls to async (`httpx.AsyncClient`) and await them.
  - For file streaming:
    - Use `StreamingResponse` for large PDFs or large audio and ensure chunked transfer.

- **Optimize heavy processing endpoints**
  - PDF endpoints that:
    - Extract text for the whole file.
    - Run multiple LLM calls (summarization + translation).
  - Consider:
    - Chunking work for very large PDFs.
    - Offloading to a background worker/queue (Celery/RQ/Arq) and returning job IDs + polling/WS endpoint.

- **Configure ASGI server for production**
  - Replace `uvicorn.run(..., host="0.0.0.0", port=...)` direct usage in `__main__` in production:
    - Use `gunicorn` with `uvicorn.workers.UvicornWorker` (or similar) or `uvicorn` run by a process manager, not from within the source module for prod.
    - Tune worker count, keep‑alive, and timeouts.

---

### 8. Data privacy and logging policy

- **Define what gets logged**
  - You are logging prompts and responses in some debug logs (`logger.debug(f"Received prompt: ...")`).
  - For production:
    - Introduce a privacy layer:
      - Mask or truncate sensitive content.
      - Put prompt logging behind a config flag.
    - Never log raw user data if it’s sensitive/PII, or hash/anonymize as needed.

- **Handle temp files and storage securely**
  - Multiple endpoints write to temp files (`NamedTemporaryFile`) and then return `FileResponse`:
    - Ensure:
      - A dedicated temp directory with proper permissions.
      - Periodic cleanup for anything not deleted via background tasks (e.g. on crash).
    - Avoid including user‑supplied filenames in paths or headers unescaped.

---

### 9. Validation, consistency, and DRYness

- **Centralize validation helpers**
  - Combine `validate_model`, `validate_language`, repeated language checks, and repeated prompt length checks into shared utilities.
  - Use Pydantic validators in request models where it makes sense:
    - Validate `model` against supported list.
    - Validate `language` fields.

- **Align business rules**
  - Time‑to‑words logic (`time_to_words`) plus Dwani system prompts:
    - Ensure one canonical behavior for queries about time.
  - Shared constants:
    - `MAX_PROMPT_LENGTH`, `MAX_TTS_TEXT_LENGTH`, etc.

- **Code hygiene**
  - Remove:
    - Duplicate imports of `FastAPI`, `UploadFile`, `File`, `HTTPException`, `BaseModel`, etc.
    - Commented out legacy blocks that are no longer used.
  - Add type hints to all helper functions for clarity.

---

### 10. Testing and quality gates

- **Unit tests**
  - For:
    - `get_openai_client` / model selection.
    - `validate_model` / `validate_language`.
    - PDF extraction helpers, OCR helpers, time formatting.

- **Integration tests**
  - Using `TestClient`:
    - Exercise each endpoint with valid and invalid inputs (wrong model, unsupported lang, oversized prompt, bad file type).
    - Mock external services (LLM, TTS, ASR, translation, PDF backends) so tests are deterministic.

- **Load and resilience tests**
  - Run:
    - Load tests against key endpoints (chat, TTS, PDF summarization).
    - Chaos tests on external dependency unavailability to see behavior (timeouts, fallback).

- **CI/CD**
  - Add:
    - Linting (ruff/flake8, black for formatting, mypy for type checking).
    - Test suite in CI.
    - Deploy pipeline that builds Docker images and deploys to your target environment.

---

### 11. Deployment and runtime setup

- **Containerize the app**
  - Create a minimal `Dockerfile`:
    - Use slim Python base.
    - Install only needed OS libs (e.g., for PDFs/vision if required).
    - Run as non‑root user.
  - Entrypoint:
    - `gunicorn "src.server.main:app" -k uvicorn.workers.UvicornWorker ...` with proper worker/threads config.

- **Put a reverse proxy in front**
  - nginx/Envoy/ALB to handle:
    - TLS termination.
    - Rate limiting / IP allowlists if required.
    - Static response caching where safe.

---

### 12. Documentation and DX

- **Clean up OpenAPI docs**
  - You already use `summary`, `description`, and tags; ensure:
    - All parameters are documented.
    - Example payloads are representative and not misleading.
  - Document:
    - Rate limits.
    - Error formats.
    - Authentication scheme.
    - Supported languages/models and how they map.

- **Versioning and change management**
  - Keep `/v1` stable; for breaking changes, introduce `/v2` and deprecate `/v1` gradually.

---
