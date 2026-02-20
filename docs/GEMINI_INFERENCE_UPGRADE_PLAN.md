# Plan: Replace Inference with Gemini in `main_all.py`

This document is a **phase-by-phase plan** to switch LLM inference in `src/server/main_all.py` from the current OpenAI-compatible backend (vLLM / `DWANI_API_BASE_URL`) to **Google Gemini** inference. The upgrade is designed so you can implement and test in phases.

---

## Current State Summary

### 1. **Client & configuration**
- **Sync**: `get_openai_client(model)` → `OpenAI(api_key="http", base_url=os.getenv("DWANI_API_BASE_URL"))`
- **Async**: `get_async_openai_client(model)` → `AsyncOpenAI(...)` with same base URL
- **Models**: Validation uses `SUPPORTED_MODELS` and per-endpoint lists: `gemma3`, `qwen3`, `sarvam-m`, `gpt-oss`, `moondream`, etc.
- **Port mapping**: Currently overridden by `DWANI_API_BASE_URL`; no per-model routing in code.

### 2. **Inference usage (all via OpenAI-style `chat.completions.create`)**

| Category | Endpoints / Functions | Sync/Async | Input type |
|----------|------------------------|------------|------------|
| **Chat (text-only)** | `/v1/indic_chat`, `/v1/chat_direct` | Sync (client used in async route) | Text messages (system + user) |
| **Translation** | `/v1/translate` (batch sentences) | Sync | Text only, JSON array output |
| **Vision / OCR** | `vision_query()`, `ocr_page_with_rolm_query()` | Sync | Image (base64) + text |
| **Visual query API** | `/v1/indic_visual_query`, `/v1/indic_visual_query_direct` | — | Call `vision_query()` |
| **OCR API** | `/v1/ocr`, `/ocr` | — | Call `ocr_page_with_rolm_query()` |
| **PDF extract (single page)** | `extract_text_from_pdf`, `extract_text_page` | Async | Image(s) + OCR prompt |
| **PDF extract (multi-page)** | `extract_text_file`, `new_extract_text_file`, `app_extract_text_from_pdf` | Async | Multiple images + text / structured output |
| **Document / custom prompt** | `indic_custom_prompt_pdf`, `indic_custom_prompt_pdf_all` | Sync + async helpers | Extracted text + chat (text-only) |
| **Custom prompt PDF** | `/v1/custom-prompt-pdf` (and similar) | Sync | Text from PDF + prompt |

### 3. **Message formats in use**
- **Text-only**: `{"role": "system"|"user", "content": string or [{"type":"text","text": "..."}]}`.
- **Multimodal**: `content` as list with `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}` and `{"type": "text", "text": "..."}`.
- **Structured output**: One place uses `response_format={ "type": "json_schema", "json_schema": {...} }` (`new_extract_text_file`).

### 4. **Dependencies**
- `openai` in `requirements.txt` (used for all current inference).

---

## Target: Gemini inference

Two main options:

1. **Google GenAI SDK (`google-genai`)**  
   - Native Gemini API: `genai.GenerativeModel(model_name).generate_content(...)`  
   - Different message/part format (e.g. `Parts` with `inline_data` for images, no `image_url`).  
   - Good for: full control, latest Gemini features, optional structured output via Gemini’s APIs.

2. **Vertex AI with OpenAI-compatible Chat Completions**  
   - Same `client.chat.completions.create(...)` interface.  
   - Swap base URL and model name; image format may still be compatible.  
   - Good for: minimal code change, reuse existing message construction.

The plan below assumes you introduce a **single abstraction layer** that can use either the current OpenAI client or Gemini (GenAI SDK or Vertex). Phases are written so you can implement “Gemini” first behind this abstraction, then retire the old client.

---

## Phase 1: Abstraction layer and configuration

**Goal:** Decide Gemini option (GenAI SDK vs Vertex), add config and a single place that returns “inference client” or “Gemini model” so the rest of the code can be switched without touching every endpoint.

### 1.1 Environment and config
- Add env vars, for example:
  - `GEMINI_API_KEY` (for Google AI / GenAI SDK) **or** Vertex: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and service-account auth.
  - Optional: `USE_GEMINI=true` / `INFERENCE_BACKEND=gemini` to switch between current backend and Gemini.
- In code: read these in one place (e.g. a small `config` module or `get_settings()` extension) and expose:
  - Whether to use Gemini.
  - Which Gemini model id to use for “chat”, “vision”, “translation” if you want different models later.

### 1.2 Inference abstraction
- Add a thin module (e.g. `src/server/llm_client.py` or `inference.py`) that exposes:
  - **Sync**: e.g. `completion_sync(messages, model=None, max_tokens=..., temperature=..., response_format=None)`  
    - Returns: `str` (content) or parsed object for structured output.
  - **Async**: e.g. `completion_async(messages, model=None, max_tokens=..., temperature=..., response_format=None)`  
    - Same return contract.
- **Message format**: Keep current “OpenAI-style” messages (list of `role` + `content` with text and/or `image_url`) inside the API server. The abstraction layer **converts** to Gemini format when `USE_GEMINI=true`:
  - For GenAI SDK: map `system` → system instruction; `user` content → `Parts`: text parts + `InlineDataPart` (mime_type + base64 data) for each image.
  - For Vertex Chat Completions: keep messages as-is and only swap client and model name.
- **Implementation**:
  - When not using Gemini: abstraction calls existing `get_openai_client(model).chat.completions.create(...)` (and async counterpart) and returns `response.choices[0].message.content` (and optional tool/parsed JSON).
  - When using Gemini: call Gemini (GenAI or Vertex) with converted request; normalize response to the same return type (string + optional structured).

### 1.3 Model naming
- Define a mapping from your current model names to Gemini model ids (e.g. `gemma3` → `gemini-2.0-flash` or `gemini-1.5-pro`) in config or in the new module.
- `validate_model()` and `SUPPORTED_MODELS` can stay for API compatibility; internally map to Gemini model id when Gemini is enabled.

**Deliverables:**  
- New config (env + optional settings).  
- New module with `completion_sync` / `completion_async` and message conversion for Gemini.  
- No change yet to route handlers; they still call `get_openai_client` / `get_async_openai_client` (or you can already switch them to the new abstraction in Phase 1 and keep backend “current” as default).

**Testing:**  
- Unit test: build a few message lists (text-only, one image + text), call abstraction with `USE_GEMINI=false` and `USE_GEMINI=true`, assert same or acceptable response shape.

---

## Phase 2: Text-only endpoints (chat and translation)

**Goal:** All text-only inference goes through the new abstraction and works with Gemini when enabled.

### 2.1 Chat endpoints
- **`/v1/indic_chat`**  
  - Replace direct `get_openai_client(chat_request.model)` + `client.chat.completions.create(...)` with a call to the new sync completion helper (or async if you add async Gemini path).  
  - Pass: system message, user message, `max_tokens`, `temperature`, model (optional).  
  - Use returned string as `generated_response` and return `ChatDirectResponse(response=generated_response)`.
- **`/v1/chat_direct`**  
  - Same change: use abstraction instead of raw OpenAI client; keep system prompt and time-to-words logic unchanged.

### 2.2 Translation endpoint
- **`/v1/translate`** (batch sentences)  
  - Replace `get_openai_client(model)` + `client.chat.completions.create(...)` with the sync completion helper.  
  - Same prompts (system + user); parse JSON array from returned string as today.  
  - Handle JSON parse errors as you do now.

### 2.3 Error handling
- In the abstraction, map Gemini-specific errors (e.g. safety, rate limit, model not found) to the same HTTP exceptions or log messages you use today so clients see consistent behavior.

**Deliverables:**  
- Chat and translation use the abstraction.  
- With `USE_GEMINI=true` and valid Gemini config, chat and translation are served by Gemini.

**Testing:**  
- Call `/v1/indic_chat`, `/v1/chat_direct`, `/v1/translate` with `USE_GEMINI=false` and `USE_GEMINI=true`; compare behavior and response format.

---

## Phase 3: Vision and single-image flows (OCR, visual query)

**Goal:** All vision/OCR and single-image endpoints use the abstraction; Gemini receives image + text correctly.

### 3.1 Helpers that do “one image + text”
- **`vision_query(img_base64, user_query, model, system_prompt)`**  
  - Today: builds messages with `image_url` + text, calls `get_openai_client(model).chat.completions.create(...)`.  
  - Change: build the same message list (for API consistency), call sync completion helper.  
  - Abstraction must support multimodal: when converting to Gemini, turn `image_url` (data URI or base64) into Gemini image part (e.g. `InlineDataPart` with mime and base64).
- **`ocr_page_with_rolm_query(img_base64, query, model)`**  
  - Same: replace direct client call with sync completion helper; ensure image part is passed through.

### 3.2 Endpoints that call these helpers
- **`/v1/indic_visual_query`**  
  - No change to request/response; it already calls `vision_query(...)`. Once `vision_query` uses the abstraction, this endpoint uses Gemini when enabled.
- **`/v1/indic_visual_query_direct`**  
  - Same (calls `indic_visual_query_direct` → eventually same vision path).
- **`/v1/ocr`** and **`/ocr`**  
  - Both end up in `ocr_image()` which uses `ocr_page_with_rolm_query`. So updating the helper is enough.

### 3.3 Image format
- In the abstraction layer, when converting to Gemini:
  - Parse `data:image/png;base64,...` or `data:image/jpeg;base64,...` from `image_url.url`.
  - Send as inline image with correct mime type and base64 payload (Gemini GenAI SDK uses `Part.from_bytes(data, mime_type="image/png")` or similar).

**Deliverables:**  
- `vision_query` and `ocr_page_with_rolm_query` use the completion abstraction.  
- All visual query and OCR endpoints work with Gemini when enabled.

**Testing:**  
- Upload a PNG, call `/v1/ocr` and `/v1/indic_visual_query` with and without Gemini; check response quality and format.

---

## Phase 4: PDF and multi-image / structured output

**Goal:** All PDF extraction and document-processing paths use the abstraction; multi-image and (where used) structured output work with Gemini.

### 4.1 Per-page PDF extraction (one image per request)
- **`extract_text_from_pdf`**  
  - Uses async client and one image per page with OCR prompt.  
  - Replace `get_openai_client(model)` and `client.chat.completions.create(...)` with async completion helper; keep the same message list (image + text).
- **`extract_text_page`**  
  - Same: async completion helper with one image + text.
- **`extract_text_file`**  
  - Loop over pages: for each page, build one image + text message, call async completion helper, concatenate results. No API change.

### 4.2 Multi-image in one request
- **`new_extract_text_file`**  
  - Sends multiple images in one user message and uses **structured output** (JSON schema).  
  - Abstraction must support:  
    - Multiple image parts in one “message” when converting to Gemini.  
    - Optional `response_format` / `response_mime_type`: for Gemini you may use response schema or `generate_content(..., generation_config=GenerationConfig(response_mime_type="application/json", ...))` and optionally a schema.  
  - Map current `response_format` (OpenAI-style json_schema) to Gemini’s equivalent (e.g. structured output or post-process JSON).  
  - Return combined text as today (e.g. `"\n\n".join(page texts)"`).
- **`app_extract_text_from_pdf`**  
  - Same as current: multiple pages, one image per call or multi-image; route through async completion helper. Ensure image parts and any structured output are supported.

### 4.3 Document / custom prompt (text-only after extraction)
- **`indic_custom_prompt_pdf`**  
  - Uses `extract_text_page` (which will use Gemini in Phase 4.1) then two **text-only** completions (answer + translation).  
  - Replace those two `get_openai_client(model).chat.completions.create(...)` calls with the sync completion helper. No message format change.
- **`indic_custom_prompt_pdf_all`**  
  - Uses `extract_text_file` then one text-only completion.  
  - Replace `get_openai_client(model).chat.completions.create(...)` with sync completion helper.

### 4.4 Other custom-prompt PDF endpoints
- **`/v1/custom-prompt-pdf`** (and any similar routes):  
  - Locate all remaining `get_openai_client(model).chat.completions.create(...)` and `get_async_openai_client(model).chat.completions.create(...)` in PDF/document flows.  
  - Replace with sync/async completion helper as appropriate.

**Deliverables:**  
- All PDF extraction and custom-prompt flows go through the abstraction.  
- Multi-image and structured output are implemented for Gemini in the abstraction.  
- No change to response models (e.g. `IndicCustomPromptPDFResponse`, `PDFTextExtractionResponse`).

**Testing:**  
- Upload a multi-page PDF; test extract-text, summarize, and custom-prompt endpoints with Gemini on/off.

---

## Phase 5: Cleanup, model list, and configuration

**Goal:** Single source of truth for “which backend and which model,” and remove dead code.

### 5.1 Model lists and validation
- When Gemini is the only backend (or default):
  - Reduce or replace `SUPPORTED_MODELS` with a Gemini-oriented list (e.g. `gemini-2.0-flash`, `gemini-1.5-pro`) or keep current names and map them in the abstraction.
  - Ensure `validate_model()` and any `valid_models` in endpoints use this list.
  - Update OpenAPI examples/defaults if they reference old model names.

### 5.2 Client removal
- Once all call sites use the abstraction:
  - Remove or deprecate direct use of `get_openai_client` and `get_async_openai_client` from `main_all.py` (or keep them only inside the abstraction when `USE_GEMINI=false`).
  - Optionally keep `DWANI_API_BASE_URL` for a “legacy” OpenAI-compatible backend and use it only inside the abstraction.

### 5.3 Env and docs
- Document new env vars (`GEMINI_API_KEY` or Vertex vars, `USE_GEMINI` / `INFERENCE_BACKEND`) in `.env.sample` and `docs/`.
  - Add a short “Gemini setup” section: API key or Vertex project, required permissions, and how to switch between backends.

### 5.4 Dependencies
- Add `google-genai` (or Vertex client) to `requirements.txt` if using GenAI SDK.  
- If you use only Vertex with OpenAI-compatible API, you may only need `openai` plus Google auth; document that.

**Deliverables:**  
- Clean model list and validation.  
- Env and README/docs updated.  
- requirements.txt updated.  
- No remaining direct `chat.completions.create` in route/helper code outside the abstraction.

---

## Implementation checklist (by phase)

- [ ] **Phase 1**  
  - [ ] Add Gemini env/config.  
  - [ ] Add `llm_client` / inference module with sync + async completion and message conversion for Gemini.  
  - [ ] Map model names to Gemini model ids.  
  - [ ] (Optional) Switch 1–2 endpoints to abstraction with backend=current to verify.

- [ ] **Phase 2**  
  - [ ] `/v1/indic_chat` → abstraction.  
  - [ ] `/v1/chat_direct` → abstraction.  
  - [ ] `/v1/translate` → abstraction.  
  - [ ] Test chat + translation with Gemini.

- [ ] **Phase 3**  
  - [ ] `vision_query` → abstraction (with image part conversion).  
  - [ ] `ocr_page_with_rolm_query` → abstraction.  
  - [ ] Test `/v1/indic_visual_query`, `/v1/indic_visual_query_direct`, `/v1/ocr`, `/ocr`.

- [ ] **Phase 4**  
  - [ ] `extract_text_from_pdf`, `extract_text_page`, `extract_text_file` → async abstraction.  
  - [ ] `new_extract_text_file`, `app_extract_text_from_pdf` → async abstraction (multi-image + structured output).  
  - [ ] `indic_custom_prompt_pdf`, `indic_custom_prompt_pdf_all` → sync abstraction for text-only steps.  
  - [ ] Any remaining custom-prompt PDF endpoints → abstraction.  
  - [ ] Test all PDF and document endpoints with Gemini.

- [ ] **Phase 5**  
  - [ ] Unify `SUPPORTED_MODELS` / `validate_model` and optional removal of direct OpenAI client usage.  
  - [ ] Update `.env.sample` and docs.  
  - [ ] Update `requirements.txt`.  
  - [ ] Final regression test (all inference paths with Gemini and, if kept, legacy backend).

---

## File-level change summary

| File / area | Changes |
|-------------|--------|
| `src/server/main_all.py` | Replace every `get_openai_client` / `get_async_openai_client` + `chat.completions.create` with calls to the new completion abstraction; keep request/response models and route signatures. |
| New: `src/server/llm_client.py` (or `inference.py`) | Abstraction: sync/async completion, message conversion (OpenAI-style → Gemini), model mapping, error mapping. |
| Config / settings | New env vars and optional “inference backend” and “Gemini model” settings. |
| `requirements.txt` | Add `google-genai` (or keep only OpenAI + Google auth for Vertex). |
| `.env.sample`, `docs/` | Document Gemini setup and env vars. |

---

## Risk and rollback

- **Feature flag:** Keeping `USE_GEMINI` or `INFERENCE_BACKEND` lets you roll back by env change without code deploy.  
- **Fallback:** You can implement “on Gemini failure, retry with current backend” in the abstraction (optional).  
- **Structured output:** Gemini’s JSON/structured support may differ from OpenAI’s `response_format`; Phase 4 should implement a clear mapping or a small adapter so `new_extract_text_file` behavior remains acceptable.

This plan gives you a clear order of work and a way to upgrade to Gemini inference in phases while keeping the rest of `main_all.py` behavior intact.
