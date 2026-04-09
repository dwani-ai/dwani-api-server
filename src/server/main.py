"""dwani.ai FastAPI application (multimodal inference API)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import logging.config
import os
import re
import tempfile
import time
from datetime import datetime
from enum import Enum
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pytz
import requests
import uvicorn
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from num2words import num2words
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    OpenAIError,
)
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# from ultralytics import YOLO
# import cv2

class Settings:
    chat_rate_limit = "10/minute"
    max_tokens = 500
    openai_api_key = "http"

def get_settings():
    return Settings()
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "simple",
            "filename": "dwani_api.log",
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["stdout", "file"],
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("indic_all_server")

# ---------------------------------------------------------------------------
# OpenAI-compatible LLM clients (DWANI_API_BASE_URL_LLM)
# ---------------------------------------------------------------------------
_VALID_LLM_MODELS = frozenset({"gemma4"})


def get_openai_client(model: str) -> OpenAI:
    """Sync client for chat completions (same backend as async)."""
    if model not in _VALID_LLM_MODELS:
        raise ValueError(
            f"Invalid model: {model}. Choose from: {', '.join(sorted(_VALID_LLM_MODELS))}"
        )
    base_url = f"{os.getenv('DWANI_API_BASE_URL_LLM')}"
    return OpenAI(api_key="http", base_url=base_url)


def get_async_openai_client(model: str) -> AsyncOpenAI:
    """Async client for vision / PDF / transcribe flows."""
    if model not in _VALID_LLM_MODELS:
        raise ValueError(
            f"Invalid model: {model}. Choose from: {', '.join(sorted(_VALID_LLM_MODELS))}"
        )
    base_url = f"{os.getenv('DWANI_API_BASE_URL_LLM')}"
    return AsyncOpenAI(api_key="http", base_url=base_url)


def encode_image(image: BytesIO) -> str:
    """Encode image bytes from a BytesIO buffer to a base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")


def _vllm_chat_completions_url() -> str:
    base = (os.getenv("DWANI_API_BASE_URL_LLM") or "").strip()
    if not base:
        raise HTTPException(
            status_code=503,
            detail="LLM backend not configured (set DWANI_API_BASE_URL_LLM)",
        )
    return f"{base.rstrip('/')}/chat/completions"


def _forward_headers_to_vllm(request: Request, body: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    ct = request.headers.get("content-type")
    if ct:
        headers["Content-Type"] = ct
    elif body:
        headers["Content-Type"] = "application/json"
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="dwani.ai API",
    description="A multimodal inference API designed for privacy.",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "https://*.hf.space",
        "https://dwani.ai",
        "https://*.dwani.ai",
        "https://dwani-*.hf.space",
        "http://localhost:11080"
        ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes: health & documentation
# ---------------------------------------------------------------------------
@app.get("/v1/health", 
         summary="Check API Health",
         description="Returns the health status of the API and the current model in use.",
         tags=["Utility"],
         response_model=dict)
async def health_check():
    return {"status": "healthy", "model": "llm_model_name"}  # Placeholder model name

@app.get("/",
         summary="Redirect to Docs",
         description="Redirects to the Swagger UI documentation.",
         tags=["Utility"])
async def home():
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Model and language constants
# ---------------------------------------------------------------------------
SUPPORTED_MODELS = ["gemma4", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]

language_options = [
    ("English", "eng_Latn"),
    ("Kannada", "kan_Knda"),
    ("Hindi", "hin_Deva"),
    ("Assamese", "asm_Beng"),
    ("Bengali", "ben_Beng"),
    ("Gujarati", "guj_Gujr"),
    ("Malayalam", "mal_Mlym"),
    ("Marathi", "mar_Deva"),
    ("Odia", "ory_Orya"),
    ("Punjabi", "pan_Guru"),
    ("Tamil", "tam_Taml"),
    ("Telugu", "tel_Telu"),
    ("German", "deu_Latn"),
]

code_to_name = {code: name for name, code in language_options}
# Superset for validation (includes codes not listed in language_options / translate UI names).
_SUPPORTED_LANG_EXTRA = frozenset(
    {
        "kas_Arab",
        "kas_Deva",
        "san_Deva",
        "brx_Deva",
        "mai_Deva",
        "sat_Olck",
        "doi_Deva",
        "snd_Arab",
        "snd_Deva",
        "gom_Deva",
        "mni_Beng",
        "mni_Mtei",
        "npi_Deva",
        "urd_Arab",
        "fra_Latn",
        "nld_Latn",
        "spa_Latn",
        "ita_Latn",
        "por_Latn",
        "rus_Cyrl",
        "pol_Latn",
    }
)
SUPPORTED_LANGUAGES = {code for _, code in language_options} | _SUPPORTED_LANG_EXTRA


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "Hello, how are you?"}}
    )

_TRANSCRIBE_TASK_PROMPT = (
    "Transcribe the audio verbatim in its native script. "
    "Output only the transcribed text. "
    "Do not translate, explain, answer questions, or add labels or commentary."
)


def _transcription_only_text(raw: str) -> str:
    s = raw.strip()
    low = s.lower()
    key = "response:"
    if key in low:
        s = s[: low.index(key)].strip()
    for line in s.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("transcription:"):
            return stripped.split(":", 1)[1].strip()
    out_lines = []
    for line in s.splitlines():
        if line.strip().lower().startswith("language:"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip() or raw.strip()


# ---------------------------------------------------------------------------
# Routes: audio (transcription)
# ---------------------------------------------------------------------------
@app.post("/v1/transcribe/", 
          response_model=TranscriptionResponse,
          summary="Transcribe Audio File",
          description="Transcribe audio via the chat completions API (gemma4 multimodal). Returns transcribed text only.",
          tags=["Audio"],
          responses={
              200: {"description": "Transcription result", "model": TranscriptionResponse},
              400: {"description": "Invalid audio"},
              504: {"description": "Transcription service timeout"}
          })
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Query(
        None,
        description="Optional legacy query param; ignored. Transcription is model-based.",
    ),
):
    start_time = time.time()
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    mime = file.content_type or "audio/wav"
    b64 = base64.standard_b64encode(file_content).decode("ascii")
    audio_data_url = f"data:{mime};base64,{b64}"

    model = "gemma4"
    try:
        client = get_async_openai_client(model)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                        {"type": "text", "text": _TRANSCRIBE_TASK_PROMPT},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=60.0,
        )
    except APITimeoutError:
        logger.error("Chat completions transcription timed out")
        raise HTTPException(status_code=504, detail="Transcription service timeout")
    except APIConnectionError as e:
        logger.error(f"Chat completions request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    except APIStatusError as e:
        err_body = getattr(e, "body", None)
        err_detail = err_body if err_body is not None else getattr(e, "message", str(e))
        logger.debug(f"Transcription error: {e.status_code} - {err_detail}")
        raise HTTPException(
            status_code=502,
            detail=f"Chat completions error: {e.status_code} {err_detail}",
        )
    except OpenAIError as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    text = ""
    if response.choices:
        msg = response.choices[0].message
        if msg and msg.content is not None:
            text = str(msg.content).strip()

    if not text:
        logger.debug("Transcription empty from chat completions")
        raise HTTPException(status_code=500, detail="Transcription failed: empty response")

    text = _transcription_only_text(text)
    if not text:
        raise HTTPException(status_code=500, detail="Transcription failed: empty response")

    logger.debug(f"Transcription completed in {time.time() - start_time:.2f} seconds")
    return TranscriptionResponse(text=text)


#model = YOLO("yolov8l.pt")  # example for large model with better accuracy


def read_imagefile(file) -> np.ndarray:
    image_bytes = file.file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image

'''
@app.post("/detect/")
async def detect(
    image_file: UploadFile = File(...),
    confidence_threshold: float = Query(0.9, gt=0, lt=1, description="Minimum confidence threshold for detections"),
    top_k: int = Query(5, gt=0, description="Maximum number of top detections to return, sorted by confidence")
):
    image = read_imagefile(image_file)
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf.cpu().numpy().item())
            class_id = int(box.cls.cpu().numpy().item())
            label = model.names[class_id]
            if confidence >= confidence_threshold:
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": class_id,
                    "label": label
                })
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:top_k]
    return {"detections": detections}

@app.post("/detect-image/")
async def detect_image(
    image_file: UploadFile = File(...),
    confidence_threshold: float = Query(0.9, gt=0, lt=1, description="Minimum confidence threshold for detections"),
    top_k: int = Query(5, gt=0, description="Maximum number of top detections to draw, sorted by confidence")
):
    image = read_imagefile(image_file)
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf.cpu().numpy().item())
            class_id = int(box.cls.cpu().numpy().item())
            label = model.names[class_id]
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "label": label
                })

    top_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:top_k]

    for det in top_detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        confidence = det["confidence"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', image)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
'''

# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic)
# ---------------------------------------------------------------------------
class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Describe the image",
                "src_lang": "eng_Latn",
                "tgt_lang": "kan_Knda",
                "model": "moondream"
            }
        }
    )


class OCRRequest(BaseModel):
    model: str = Field(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "moondream"
            }
        }
    )


class VisualQueryDirectRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    model: str = Field(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Describe the image",
                "model": "moondream"
            }
        }
    )


class VisualQueryResponse(BaseModel):
    answer: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"answer": "The image shows a screenshot of a webpage."}}
    )


class OCRResponse(BaseModel):
    answer: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"answer": "The image shows a screenshot of a webpage."}}
    )


class VisualQueryDirectResponse(BaseModel):
    answer: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"answer": "The image shows a screenshot of a webpage."}}
    )


class PDFTextExtractionResponse(BaseModel):
    page_content: str = Field(..., description="Extracted text from the specified PDF page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page_content": "Google Interview Preparation Guide\nCustomer Engineer Specialist\n\nOur hiring process\n..."
            }
        }
    )


class PDFTextExtractionAllResponse(BaseModel):
    page_contents: Dict[str, str] = Field(
        ..., description="Extracted text from each PDF page, with page numbers as keys and text content as values"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page_contents": {
                    "0": "Google Interview Preparation Guide\nCustomer Engineer Specialist\n\nOur hiring process\n...",
                    "1": "Page 2 content\nAdditional details about the interview process\n..."
                }
            }
        }
    )


class DocumentProcessPage(BaseModel):
    processed_page: int = Field(..., description="Page number of the extracted text")
    page_content: str = Field(..., description="Extracted text from the page")
    translated_content: Optional[str] = Field(None, description="Translated text of the page, if applicable")


class DocumentProcessResponse(BaseModel):
    pages: List[DocumentProcessPage] = Field(..., description="List of pages with extracted and translated text")


class SummarizePDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    summary: str = Field(..., description="Summary of the specified page")
    processed_page: int = Field(..., description="Page number processed")


class IndicSummarizePDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    summary: str = Field(..., description="Summary of the specified page in the source language")
    translated_summary: str = Field(..., description="Summary translated into the target language")
    processed_page: int = Field(..., description="Page number processed")


class IndicSummarizeAllPDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    summary: str = Field(..., description="Summary of the specified page in the source language")
    translated_summary: str = Field(..., description="Summary translated into the target language")


class CustomPromptPDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    response: str = Field(..., description="Response based on the custom prompt")
    processed_page: int = Field(..., description="Page number processed")


class IndicCustomPromptPDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    query_answer: str = Field(..., description="Response based on the custom prompt")
    translated_query_answer: str = Field(..., description="Translated response in the target language")
    processed_page: int = Field(..., description="Page number processed")


class IndicCustomPromptPDFAllResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    query_answer: str = Field(..., description="Response based on the custom prompt")
    translated_query_answer: str = Field(..., description="Translated response in the target language")


# Helper function for model selection
def validate_model(model: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}. Must be one of {SUPPORTED_MODELS}")
    return model


def validate_language(lang: str, field_name: str) -> str:
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {lang}. Must be one of {SUPPORTED_LANGUAGES}")
    return lang


class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "Hi there, I'm doing great!"}}
    )


class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

    model_config = ConfigDict(
        json_schema_extra={"example": {"result": "Processed audio output"}}
    )


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma4", description="LLM model")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Hello, how are you?",
                "src_lang": "kan_Knda",
                "tgt_lang": "kan_Knda",
                "model": "gemma4"
            }
        }
    )


class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    model: str = Field(default="gemma4", description="LLM model")
    system_prompt: str = Field(default="", description="System prompt")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Hello, how are you?",
                "model": "gemma4",
                "system_prompt": ""
            }
        }
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    model_config = ConfigDict(
        json_schema_extra={"example": {"response": "Hi there, I'm doing great!"}}
    )


class ChatDirectResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    model_config = ConfigDict(
        json_schema_extra={"example": {"response": "Hi there, I'm doing great!"}}
    )


class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sentences": ["Hello", "How are you?"],
                "src_lang": "en",
                "tgt_lang": "kan_Knda"
            }
        }
    )


class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    model_config = ConfigDict(
        json_schema_extra={"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}}
    )


# ---------------------------------------------------------------------------
# Routes: audio (TTS / speech)
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech",
          summary="Generate Speech from Text",
          description="Convert text to speech using an external TTS service and return as a downloadable audio file.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio file", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid or empty input"},
              502: {"description": "External TTS service unavailable"},
              504: {"description": "TTS service timeout"}
          })
async def generate_audio(
    request: Request,
    input: str = Query(..., description="Text to convert to speech (max 10000 characters)"),
    response_format: str = Query("mp3", description="Audio format (ignored, defaults to mp3 for external API)"),
    language: str = Query("kannada", description="language for TTS"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if len(input) > 10000:
        raise HTTPException(status_code=400, detail="Input cannot exceed 10000 characters")
    
    logger.debug("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(input),
        "client_ip": request.client.host
    })
    

        # Validate language
    allowed_languages = ["kannada", "hindi", "tamil", "english","german", "telugu", "marathi" ]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    start_time = time.time()
   
    if( language in ["english", "german"]):
        openai = OpenAI(base_url="http://localhost:8000/v1", api_key="cant-be-empty")
        model_id = "speaches-ai/Kokoro-82M-v1.0-ONNX"
        voice_id = "af_heart"

        # Create speech
        res = openai.audio.speech.create(
            model=model_id,
            voice=voice_id,
            input=input,
            response_format="wav",
            speed=1,
        )
# Prepare headers for the response
        headers = {
            "Content-Disposition": "attachment; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
        }
        # Save the audio to a file
        output_file = Path("output.wav")
        with output_file.open("wb") as f:
            f.write(res.response.read())

        return FileResponse(
            path=output_file,
            filename="speech.mp3",
            media_type="audio/mp3",
            headers=headers
        )
    else:    
            
        payload = {"text": input}
        
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file_path = temp_file.name
        
        try:
            base_url = f"{os.getenv('DWANI_API_BASE_URL_TTS')}/v1/audio/speech"
            response = requests.post(
                base_url,
                json=payload,
                headers={"accept": "*/*", "Content-Type": "application/json"},
                stream=True,
                timeout=30
            )
            
            # Write audio content to the temporary file
            with open(temp_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Prepare headers for the response
            headers = {
                "Content-Disposition": "attachment; filename=\"speech.mp3\"",
                "Cache-Control": "no-cache",
            }
            
            # Schedule file cleanup as a background task
            def cleanup_file(file_path: str):
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        logger.debug(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary file {file_path}: {str(e)}")
            
            background_tasks.add_task(cleanup_file, temp_file_path)
            
            # Return the file as a FileResponse
            return FileResponse(
                path=temp_file_path,
                filename="speech.mp3",
                media_type="audio/mp3",
                headers=headers
            )
            
        except requests.HTTPError as e:
            logger.error(f"External TTS request failed: {str(e)}")
            raise HTTPException(status_code=502, detail=f"External TTS service error: {str(e)}")
        finally:
            # Close the temporary file to ensure it's fully written
            temp_file.close()


# ---------------------------------------------------------------------------
# Routes: chat
# ---------------------------------------------------------------------------
@app.post(
    "/v1/chat/completions",
    tags=["Chat"],
    summary="OpenAI-compatible chat completions (vLLM proxy)",
    description=(
        "Forwards the raw request body to the configured OpenAI-compatible "
        "`POST .../v1/chat/completions` endpoint (DWANI_API_BASE_URL_LLM). "
        'When the JSON body sets `"stream": true`, the upstream SSE stream is relayed.'
    ),
    response_model=None,
    responses={
        200: {"description": "Chat completion JSON or streamed events"},
        502: {"description": "Upstream LLM error"},
        503: {"description": "LLM URL not configured"},
        504: {"description": "Upstream timeout"},
    },
)
async def openai_chat_completions_proxy(request: Request):
    body = await request.body()
    upstream = _vllm_chat_completions_url()
    fwd_headers = _forward_headers_to_vllm(request, body)

    stream = False
    if body:
        try:
            stream = json.loads(body).get("stream") is True
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    timeout = httpx.Timeout(600.0, connect=30.0)

    try:
        if stream:
            client = httpx.AsyncClient(timeout=timeout)
            stream_ctx = client.stream("POST", upstream, content=body, headers=fwd_headers)
            try:
                upstream_resp = await stream_ctx.__aenter__()
            except Exception:
                await client.aclose()
                raise
            status_code = upstream_resp.status_code
            media_type = upstream_resp.headers.get(
                "content-type", "text/event-stream; charset=utf-8"
            )

            async def streamed():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await stream_ctx.__aexit__(None, None, None)
                    await client.aclose()

            return StreamingResponse(
                streamed(), status_code=status_code, media_type=media_type
            )

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream, content=body, headers=fwd_headers)
    except httpx.TimeoutException:
        logger.error("vLLM chat/completions proxy timed out")
        raise HTTPException(status_code=504, detail="Upstream LLM timeout")
    except httpx.RequestError as e:
        logger.error(f"vLLM chat/completions proxy request error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream LLM unreachable: {e}")

    media_type = resp.headers.get("content-type", "application/json")
    return Response(
        content=resp.content, status_code=resp.status_code, media_type=media_type
    )


@app.post("/v1/indic_chat",
          response_model=ChatResponse,
          summary="Chat with AI",
          description="Generate a chat response from a prompt, language code, and model, with translation support and time-to-words conversion.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatResponse},
              400: {"description": "Invalid prompt, language code, or model"},
              504: {"description": "Chat service timeout"}
          })
async def chat_v2(
    request: Request,
    chat_request: ChatRequest
):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 10000 characters")

    # Validate model parameter
    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}, model: {chat_request.model}")

    valid_models = ["gemma4", "qwen3", "sarvam-m", "gpt-oss"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    settings = get_settings()

    logger.debug(f"Received prompt: {chat_request.prompt},  model: {chat_request.model}")

    language_name = get_language_name(chat_request.tgt_lang)

    system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Do not explain .  Return answer only in {language_name}" 

    try:
        prompt_to_process = chat_request.prompt

        client = get_openai_client(chat_request.model)
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt }]
                
                },
                {"role": "user", "content": [{"type": "text", "text": prompt_to_process}]}
            ],
            temperature=0.3,
            max_tokens=settings.max_tokens
        )
        generated_response = response.choices[0].message.content
        logger.debug(f"Generated response: {generated_response}")

        return ChatDirectResponse(response=generated_response)

    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def time_to_words():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    hour = now.hour % 12 or 12
    minute = now.minute
    hour_word = num2words(hour, to='cardinal')
    if minute == 0:
        return f"{hour_word} o'clock"
    else:
        minute_word = num2words(minute, to='cardinal')
        return f"{hour_word} hours and {minute_word} minutes"


@app.post("/v1/chat_direct",
          response_model=ChatDirectResponse,
          summary="Chat with AI",
          description="Generate a chat response from a prompt,model",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatDirectResponse},
              400: {"description": "Invalid prompt or model"},
              504: {"description": "Chat service timeout"}
          })
async def chat_direct(
    request: Request,
    chat_request: ChatDirectRequest
):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 10000 characters")

    # Validate model parameter
    valid_models = ["gemma4", "qwen3", "sarvam-m", "gpt-oss"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    settings = get_settings()

    logger.debug(f"Received prompt: {chat_request.prompt},  model: {chat_request.model}")

    try:
        prompt_to_process = chat_request.prompt

        system_prompt = chat_request.system_prompt

        current_time = time_to_words()

        dwani_prompt = f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}" 
        client = get_openai_client(chat_request.model)
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=[
                {
                    "role": "system",
                #    "content": [{"type": "text", "text": f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]
                    "content": [{"type": "text", "text": system_prompt }]
                
                },
                {"role": "user", "content": [{"type": "text", "text": prompt_to_process}]}
            ],
            temperature=0.3,
            max_tokens=settings.max_tokens
        )
        generated_response = response.choices[0].message.content
        logger.debug(f"Generated response: {generated_response}")

        return ChatDirectResponse(response=generated_response)

    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# ---------------------------------------------------------------------------
# Routes: translation
# ---------------------------------------------------------------------------
@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of sentences via the configured LLM (same backend as chat; DWANI_API_BASE_URL_LLM).",
          tags=["Translation"],
          responses={
              200: {"description": "Translation result", "model": TranslationResponse},
              400: {"description": "Invalid sentences or languages"},
              500: {"description": "Translation service error"},
              504: {"description": "Translation service timeout"}
          })
async def translate(
    request: TranslationRequest
):
    # Validate inputs
    if not request.sentences:
        raise HTTPException(status_code=400, detail="Sentences cannot be empty")
    
    # Validate language codes
    if request.src_lang not in SUPPORTED_LANGUAGES or request.tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language codes: src={request.src_lang}, tgt={request.tgt_lang}")

    # Convert language codes to names for the prompt
    src_name = code_to_name.get(request.src_lang, request.src_lang)
    tgt_name = code_to_name.get(request.tgt_lang, request.tgt_lang)

    logger.debug(f"Received translation request: {len(request.sentences)} sentences, src_lang: {request.src_lang} ({src_name}), tgt_lang: {request.tgt_lang} ({tgt_name})")

    model = "gemma4"
    client = get_openai_client(model)

    system_prompt = f"You are a professional translator. Translate the following list of sentences from {src_name} to {tgt_name}. Respond ONLY with a valid JSON array of the translated sentences in the same order, without any additional text or explanations."
    
    sentences_text = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(request.sentences)])
    user_prompt = f"Sentences to translate:\n\n{sentences_text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent translations
            max_tokens=2000   # Adjust based on expected output length
        )

        query_answer = response.choices[0].message.content.strip()

        # Parse and normalize the JSON array from the response into List[str]
        try:
            data = json.loads(query_answer)
        except json.JSONDecodeError:
            # Fallback: treat whole response as single translation string
            logger.warning(f"Non-JSON response from translation model, using raw text: {query_answer}")
            translations = [query_answer]
        else:
            translations = []
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    translations = data
                elif all(isinstance(item, dict) for item in data):
                    possible_keys = ["translation", "translated", "tgt", "text", "tr"]
                    for item in data:
                        value = None
                        for key in possible_keys:
                            if key in item and isinstance(item[key], str):
                                value = item[key]
                                break
                        if value is None:
                            logger.warning(f"Could not extract translation from item: {item}")
                            raise HTTPException(status_code=500, detail="Invalid response format from translation model")
                        translations.append(value)
                else:
                    logger.warning(f"Unexpected item types in response: {data}")
                    raise HTTPException(status_code=500, detail="Invalid response format from translation model")
            elif isinstance(data, str):
                # Model returned a plain string, wrap it
                translations = [data]
            elif isinstance(data, dict) and "translations" in data and isinstance(data["translations"], list):
                inner = data["translations"]
                if all(isinstance(item, str) for item in inner):
                    translations = inner
                else:
                    logger.warning(f"Unexpected item types in 'translations' field: {inner}")
                    raise HTTPException(status_code=500, detail="Invalid response format from translation model")
            else:
                logger.warning(f"Unexpected response format (not list/str/dict[translations]): {data}")
                raise HTTPException(status_code=500, detail="Invalid response format from translation model")

        if len(translations) != len(request.sentences):
            logger.warning(f"Unexpected response length: {translations}")
            raise HTTPException(status_code=500, detail="Invalid response format from translation model")

        logger.debug(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)

    except HTTPException:
        # Re-raise HTTPExceptions we raised above without wrapping
        raise
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


def get_language_name(lang_code):
    for name, code in language_options:
        if code == lang_code:
            return name
    return "English"


# ---------------------------------------------------------------------------
# Routes: vision / visual query
# ---------------------------------------------------------------------------
@app.post("/v1/indic_visual_query",
          response_model=VisualQueryResponse,
          summary="Visual Query with Image",
          description="Process a visual query with a text query, image, and language codes. Provide query as form data, image as file upload, and source/target languages and model as query parameters.",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": VisualQueryResponse},
              400: {"description": "Invalid query, image, or language codes"},
              422: {"description": "Validation error in request body"},
              504: {"description": "Visual query service timeout"}
          })
async def visual_query(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image (e.g., 'describe the image')"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    src_lang: str = Query(..., description="Source language code (e.g., eng_Latn, kan_Knda)"),
    tgt_lang: str = Query(..., description="Target language code (e.g., eng_Latn, kan_Knda)"),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    # Validate query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 10000 characters")


    if src_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}. Must be one of {SUPPORTED_LANGUAGES}")
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {tgt_lang}. Must be one of {SUPPORTED_LANGUAGES}")

    # Validate model
    validate_model(model)

    logger.debug("Processing visual query request", extra={
        "endpoint": "/v1/indic_visual_query",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": request.client.host,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model
    })

    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    logger.debug(f"Processing indic visual query: model={model}, prompt={query[:50] if query else None}")

    image_bytes = await file.read()
    image = BytesIO(image_bytes)
    img_base64 = encode_image(image)
    
    language_name = get_language_name(tgt_lang)

    system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Do not explain .  Return answer only in {language_name}" 

    extracted_text = vision_query(img_base64, query, model, system_prompt=system_prompt)

    response = extracted_text

    result = {
        "extracted_text": extracted_text,
        "response": response
    }
    if response:
        result["response"] = response

    logger.debug(f"visual query direct successful: extracted_text_length={len(extracted_text)}")
    return VisualQueryResponse(answer=extracted_text)


# Visual Query Endpoint
@app.post("/v1/visual_query_direct",
          response_model=VisualQueryDirectResponse,
          summary="Visual Query with Image",
          description="Process a visual query with a text query, image.Provide query as form data, image as file upload and model as query parameters.",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": VisualQueryDirectResponse},
              400: {"description": "Invalid query, image"},
              422: {"description": "Validation error in request body"},
              504: {"description": "Visual query service timeout"}
          })
async def visual_query_direct(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image (e.g., 'describe the image')"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    # Validate query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 10000 characters")

 
    # Validate model
    validate_model(model)

    logger.debug("Processing visual query direct request", extra={
        "endpoint": "/v1/visual_query_direct",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": request.client.host,
        "model": model
    })

    try:
        response = await indic_visual_query_direct(file=file, prompt=query, model=model)
        # If response is a JSONResponse, extract and parse its body
        if isinstance(response, JSONResponse):
            response_body = json.loads(response.body.decode("utf-8"))
            answer = response_body.get("response", "")
        else:
            # If response is already a dict, access it directly
            answer = response.get("response", "")

        # Continue with your logic
#        return {"answer": answer}


        if not answer:
            logger.warning(f"Empty or missing 'response' field in external API response: {answer}")
            raise HTTPException(status_code=500, detail="No valid response provided by visual query direct service")

        logger.debug(f"Visual query direct successful: {answer}")
        return VisualQueryResponse(answer=answer)

    except requests.Timeout:
        logger.error("Visual query direct request timed out")
        raise HTTPException(status_code=504, detail="Visual query direct service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query direct failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from visual query direct service")


class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

@app.post("/v1/speech_to_speech",
          summary="Speech-to-Speech Conversion",
          description="Convert input speech to processed speech in the specified language by calling an external speech-to-speech API.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid input or language"},
              504: {"description": "External API timeout"},
              500: {"description": "External API error"}
          })
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil)")
) -> StreamingResponse:
    # Validate language
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    logger.debug("Processing speech-to-speech request", extra={
        "endpoint": "/v1/speech_to_speech",
        "audio_filename": file.filename,
        "language": language,
        "client_ip": request.client.host
    })

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        external_url = f"{os.getenv('DWANI_API_BASE_URL_S2S')}/v1/speech_to_speech?language={language}"

        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        headers = {
            "Content-Disposition": f"inline; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
            "Content-Type": "audio/mp3"
        }

        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type="audio/mp3",
            headers=headers
        )

    except requests.Timeout:
        logger.error("External speech-to-speech API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External speech-to-speech API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")


# ---------------------------------------------------------------------------
# Routes: PDF (extract, summarize, custom prompts)
# ---------------------------------------------------------------------------
@app.post("/v1/extract-text",
          response_model=PDFTextExtractionResponse,
          summary="Extract Text from PDF",
          description="Extract text from a specified page of a PDF file by calling an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text", "model": PDFTextExtractionResponse},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def extract_text(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    page_number: int = Query(1, description="Page number to extract text from (1-based indexing)", ge=1),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)

    logger.debug("Processing PDF text extraction request", extra={
        "endpoint": "/v1/extract-text",
        "file_name": file.filename,
        "page_number": page_number,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/extract-text/"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {"page_number": page_number, "model": model}
        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        extracted_text = response_data.get("page_content", "")
        if not extracted_text:
            logger.warning("No page_content found in external API response")
            extracted_text = ""

        logger.debug(f"PDF text extraction completed in {time.time() - start_time:.2f} seconds")
        return PDFTextExtractionResponse(page_content=extracted_text.strip())

    except requests.Timeout:
        logger.error("External PDF extraction API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External PDF extraction API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")
    

@app.post("/v1/extract-text-all",
          response_model=PDFTextExtractionAllResponse,
          summary="Extract Text from PDF",
          description="Extract text from a specified page of a PDF file by calling an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text", "model": PDFTextExtractionAllResponse},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def extract_text_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    validate_model(model)

    logger.debug("Processing PDF text extraction", extra={
        "endpoint": "/v1/extract-text-all",
        "file_name": file.filename,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/extract-text-all/"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {"model": model}

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30

        )
        response.raise_for_status()

        response_data = response.json()
        
        # Validate response using Pydantic model
        try:
            validated_response = PDFTextExtractionAllResponse(**response_data)
            extracted_text = validated_response.page_contents
        except Exception as e:
            logger.warning(f"Failed to validate response with Pydantic model: {str(e)}")
            # Fallback to directly accessing page_contents
            extracted_text = response_data.get("page_contents", {})
        
        if not extracted_text:
            logger.warning("No page_contents found in external API response")
            extracted_text = {}

        logger.debug(f"PDF text extraction completed in {time.time() - start_time:.2f} seconds")
        return PDFTextExtractionAllResponse(page_contents=extracted_text)

    except requests.Timeout:
        logger.error("External PDF extraction API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External PDF extraction API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")


@app.post("/v1/extract-text-all-chunk",
          response_model=PDFTextExtractionAllResponse,
          summary="Extract Text from PDF",
          description="Extract text from a specified page of a PDF file by calling an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text", "model": PDFTextExtractionAllResponse},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def extract_text_all_chunk(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    validate_model(model)

    logger.debug("Processing PDF text extraction", extra={
        "endpoint": "/v1/extract-text-all-chunk",
        "file_name": file.filename,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/extract-text-all-chunk/"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {"model": model}

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30

        )
        response.raise_for_status()

        response_data = response.json()
        
        # Validate response using Pydantic model
        try:
            validated_response = PDFTextExtractionAllResponse(**response_data)
            extracted_text = validated_response.page_contents
        except Exception as e:
            logger.warning(f"Failed to validate response with Pydantic model: {str(e)}")
            # Fallback to directly accessing page_contents
            extracted_text = response_data.get("page_contents", {})
        
        if not extracted_text:
            logger.warning("No page_contents found in external API response")
            extracted_text = {}

        logger.debug(f"PDF text extraction completed in {time.time() - start_time:.2f} seconds")
        return PDFTextExtractionAllResponse(page_contents=extracted_text)

    except requests.Timeout:
        logger.error("External PDF extraction API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External PDF extraction API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")


# Indic Extract Text Endpoint
@app.post("/v1/indic-extract-text/",
          response_model=DocumentProcessResponse,
          summary="Extract and Translate Text from PDF",
          description="Extract text from a PDF page and translate it into the target language.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted and translated text", "model": DocumentProcessResponse},
              400: {"description": "Invalid PDF, page number, or language codes"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def extract_and_translate(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    src_lang: str = Form("eng_Latn", description="Source language code (e.g., eng_Latn)"),
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid language codes: src={src_lang}, tgt={tgt_lang}")

    validate_model(model)

    logger.debug("Processing indic extract text request", extra={
        "endpoint": "/v1/indic-extract-text",
        "file_name": file.filename,
        "page_number": page_number,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-extract-text/"
    #start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
            "page_number": page_number,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "model": model
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        page_content = response_data.get("page_content", "")
        translated_content = response_data.get("translated_content", "")
        processed_page = response_data.get("processed_page", page_number)

        page = DocumentProcessPage(
            processed_page=processed_page,
            page_content=page_content,
            translated_content=translated_content
        )

        #logger.debug(f"Indic extract text completed in {time.time() - start_time:.2f} seconds")
        return DocumentProcessResponse(pages=[page])

    except requests.Timeout:
        logger.error("External indic extract text API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External indic extract text API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")
    finally:
        await file.close()

# Summarize PDF Endpoint
@app.post("/v1/summarize-pdf",
          response_model=SummarizePDFResponse,
          summary="Summarize a Specific Page of a PDF",
          description="Summarize the content of a specific page of a PDF file using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text and summary", "model": SummarizePDFResponse},
              400: {"description": "Invalid PDF or page number"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def summarize_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    page_number: int = Form(..., description="Page number to summarize (1-based indexing)", ge=1),
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)

    logger.debug("Processing PDF summary request", extra={
        "endpoint": "/v1/summarize-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/summarize-pdf"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {"page_number": page_number, "model": model}

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        original_text = response_data.get("original_text", "")
        summary = response_data.get("summary", "")
        processed_page = response_data.get("processed_page", page_number)

        if not original_text or not summary:
            logger.warning(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, summary={'present' if summary else 'missing'}")
            return SummarizePDFResponse(
                original_text=original_text or "No text extracted",
                summary=summary or "No summary provided",
                processed_page=processed_page
            )

        logger.debug(f"PDF summary completed in {time.time() - start_time:.2f} seconds")
        return SummarizePDFResponse(
            original_text=original_text,
            summary=summary,
            processed_page=processed_page
        )

    except requests.Timeout:
        logger.error("External PDF summary API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External PDF summary API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")

# Indic Summarize PDF Endpoint (Updated)
@app.post("/v1/indic-summarize-pdf",
          response_model=IndicSummarizePDFResponse,
          summary="Summarize and Translate a Specific Page of a PDF",
          description="Summarize the content of a specific page of a PDF file and translate the summary into the target language using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, summary, and translated summary", "model": IndicSummarizePDFResponse},
              400: {"description": "Invalid PDF, page number, or language codes"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def indic_summarize_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    page_number: int = Form(..., description="Page number to summarize (1-based indexing)", ge=1),
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),  # Default added
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    logger.debug(f"Processing indic summarize PDF: page_number={page_number}, model={model}, tgt_lang={tgt_lang} and file={file.filename}")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    # Validate inputs
    validate_model(model)
    validate_language(tgt_lang, "target language")

    logger.debug("Processing Indic PDF summary request", extra={
        "endpoint": "/v1/indic-summarize-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "tgt_lang": tgt_lang,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-summarize-pdf"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
            "page_number": page_number,
            "tgt_lang": tgt_lang,
            "model": model
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        original_text = response_data.get("original_text", "")
        summary = response_data.get("summary", "")
        translated_summary = response_data.get("translated_summary", "")
        processed_page = response_data.get("processed_page", page_number)

        if not original_text or not summary or not translated_summary:
            logger.debug(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, summary={'present' if summary else 'missing'}, translated_summary={'present' if translated_summary else 'missing'}")
            return IndicSummarizePDFResponse(
                original_text=original_text or "No text extracted",
                summary=summary or "No summary provided",
                translated_summary=translated_summary or "No translated summary provided",
                processed_page=processed_page
            )

        logger.debug(f"Indic PDF summary completed in {time.time() - start_time:.2f} seconds, page processed: {processed_page}")
        return IndicSummarizePDFResponse(
            original_text=original_text,
            summary=summary,
            translated_summary=translated_summary,
            processed_page=processed_page
        )

    except requests.Timeout:
        logger.error("External Indic PDF summary API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External Indic PDF summary API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")


async def get_base64_msg_from_pdf(file):
    try:
        images = await render_pdf_to_png(file)
    except Exception as e:
        logger.error(f"Failed to render PDF to PNG: {str(e)}")
        return []

    messages = []
    for i, image in enumerate(images):
        try:
            # Ensure the image is in RGB mode (required for JPEG)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Save image to BytesIO as JPEG
            image_bytes_io = BytesIO()
            image.save(image_bytes_io, format="JPEG", quality=85)
            image_bytes_io.seek(0)
            
            # Encode to base64
            image_bytes = image_bytes_io.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Validate base64 string
            try:
                base64.b64decode(image_base64, validate=True)
            except Exception as e:
                logger.error(f"Invalid base64 string for page {i}: {str(e)}")
                continue
            
            # Create message (adjust based on vLLM's expected format)
            messages.append({
                "type": "image_url",
                # Option 1: Include data URI (if vLLM supports it)
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                # Option 2: Raw base64 string (uncomment if vLLM expects this)
                # "image_url": {"url": image_base64}
            })
        except Exception as e:
            logger.error(f"Image processing failed for page {i}: {str(e)}")
            continue
    
    return messages


async def render_pdf_to_png(pdf_file):

    # Save uploaded file temporarily
    try:
        with open("temp.pdf", "wb") as f:
            f.write(await pdf_file.read())
        images = convert_from_path("temp.pdf")
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF to images: {str(e)}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

    return images


def sanitize_json_string(s: str) -> str:
    """Sanitize a string to ensure it is valid for JSON parsing."""
    if not s:
        return "{}"  # Return empty JSON object if input is empty
    # Replace control characters with escaped Unicode
    s = re.sub(r'[\x00-\x1F\x7F]', lambda m: '\\u{:04x}'.format(ord(m.group())), s)
    # Remove newlines/tabs outside of string values (before/after braces, brackets, etc.)
    s = re.sub(r'[\n\t]+(?=[\{\[\]\},:0-9])', ' ', s)
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    # Ensure the string starts with a valid JSON structure
    s = s.strip()
    if not s.startswith('{') and not s.startswith('['):
        s = '{' + s + '}'
    return s

async def extract_text_batch_from_pdf(
    file: UploadFile = File(...),
    model: str = Body("gemma4", embed=True)
) -> JSONResponse:
    """Extract text from all PDF pages in a single batch request."""
    temp_file_path = None
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")
        
        messages = await get_base64_msg_from_pdf(file)
        num_pages = len(messages)
        messages.append({
            "type": "text",
            "text": (
                f"Extract plain text from these {num_pages} PDF pages. "
                "Return the results as a valid JSON object where keys are page numbers (starting from 0) "
                "and values are the extracted text for each page. "
                "Ensure the response is strictly JSON-formatted with no markdown, code blocks, or additional text outside the JSON object. "
                "Escape all special characters (e.g., newlines, tabs) properly within JSON string values to ensure valid JSON parsing. "
                "Example: {\"0\": \"Page text\\nwith newlines escaped\", \"1\": \"Another page\"}"
            )
        })

        try:
            client = get_openai_client(model)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": messages}],
                temperature=0.2,
                max_tokens=50000  # Increased to handle large PDFs
            )
            
            raw_response = response.choices[0].message.content
            logger.debug("Raw OCR response length: %d, content: %s", len(raw_response), raw_response[:500])
            
            # Clean markdown code blocks
            cleaned_response = re.sub(r'^```(?:json)?\n|\n```$', '', raw_response, flags=re.MULTILINE).strip()
            logger.debug("Cleaned response before sanitization: %s", cleaned_response[:500])
            
            # Sanitize the response
            cleaned_response = sanitize_json_string(cleaned_response)
            logger.debug("Sanitized response: %s", cleaned_response[:500])
            
            try:
                page_contents = json.loads(cleaned_response)
                logger.debug("Parsed page contents: %s", page_contents)
                if not isinstance(page_contents, dict):
                    raise ValueError("Response is not a valid JSON object with page numbers as keys")
            except json.JSONDecodeError as e:
                logger.error("JSON parsing failed: %s. Attempting fallback parsing.", str(e))
                # Fallback: Try to parse up to the last valid JSON object
                try:
                    cleaned_response = cleaned_response[:cleaned_response.rfind('}')+1]
                    page_contents = json.loads(cleaned_response)
                    logger.debug("Fallback parsing succeeded: %s", page_contents)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse OCR response as JSON after fallback: {str(e)}. Raw response: {cleaned_response[:500]}"
                    )

            return JSONResponse(content={"page_contents": page_contents})

        except Exception as e:
            logger.error("OCR batch processing failed: %s", str(e))
            raise HTTPException(status_code=500, detail=f"OCR batch processing failed: {str(e)}")

    except Exception as e:
        logger.error("Error in extract_text_batch_from_pdf: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def extract_text_from_pdf(file: UploadFile = File(...), model: str = Body("gemma4", embed=True)) -> JSONResponse:
    """Extract text from all PDF pages one at a time."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")

        validate_model(model)  # Validate model
        ocr_query_string = "Return the plain text extracted from this image."

        # Read PDF and convert to base64 images
        pages = await get_base64_msg_from_pdf(file)
        page_contents = {}

        client = get_openai_client(model)
        
        for page_num, base64_image in enumerate(pages):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                },
                                {"type": "text", "text": ocr_query_string}
                            ]
                        }
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                
                text = response.choices[0].message.content
                if not text.strip():
                    logger.warning(f"No text extracted for page {page_num}")
                    page_contents[str(page_num)] = ""
                else:
                    page_contents[str(page_num)] = text
                
            #except openai.OpenAIError as e:
            #    logger.error(f"OpenAI API error for page {page_num}: {str(e)}")
            #    page_contents[str(page_num)] = ""
            except Exception as e:
                logger.error(f"Unexpected error processing page {page_num}: {str(e)}")
                page_contents[str(page_num)] = ""

        return JSONResponse(content={"page_contents": page_contents})

    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await file.close()


async def extract_text_file(pdf_file):
    model="gemma4"
    client = get_async_openai_client(model)
    images = await render_pdf_to_png(pdf_file)
    result = ""
    for image in images:
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG', quality=85)
        image_bytes_io.seek(0)
        image_base64 = encode_image(image_bytes_io)
        
        single_message = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
            {
                "type": "text",
                "text": (
                    f"Extract plain text from this single PDF page "
                )
            }
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": single_message}],
            temperature=0.2,
            max_tokens=2048
        )
        raw_response = response.choices[0].message.content
        result = result + " " + raw_response
    
    return result

async def extract_text_page(pdf_file, page_number):
    model="gemma4"
    client = get_async_openai_client(model)
    images = await render_pdf_to_png(pdf_file)
    
    image_index = page_number-1

    image_parse = images[image_index]
    image_bytes_io = BytesIO()
    image_parse.save(image_bytes_io, format='JPEG', quality=85)
    image_bytes_io.seek(0)
    image_base64 = encode_image(image_bytes_io)
    
    single_message = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        },
        {
            "type": "text",
            "text": (
                f"Extract plain text from this single PDF page "
            )
        }
    ]
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": single_message}],
        temperature=0.2,
        max_tokens=2048
    )
    raw_response = response.choices[0].message.content
    #result = result + " " + raw_response
    
    return raw_response


# ---------------------------------------------------------------------------
# App PDF text extraction (upload helper)
# ---------------------------------------------------------------------------
class ExtractionResponse(BaseModel):
    extracted_text: str
    page_count: int
    status: str = "success"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status: str = "error"


async def app_extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """
    Core extraction logic based on your original async function.
    Processes PDF pages as images and extracts text using a vision model.
    """
    model = "gemma4"  # or whatever vision model you're using (e.g., gpt-4o, gemma4, etc.)
    client = get_async_openai_client(model)
    
    # Convert PDF pages to images
    images: List[Image.Image] = await render_pdf_to_png(pdf_file)
    
    if not images:
        raise HTTPException(status_code=400, detail="No pages found in PDF or failed to render pages")

    result = ""
    
    for i, image in enumerate(images):
        image_bytes_io = BytesIO()
        image.save(image_bytes_io, format='JPEG', quality=85)
        image_bytes_io.seek(0)
        image_base64 = encode_image(image_bytes_io)  # or base64.b64encode(...).decode()

        single_message = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            },
            {
                "type": "text",
                "text": f"Extract plain text from this single PDF page (page {i+1}). "
                        "Preserve the original reading order, headings, lists, and paragraph structure as much as possible. "
                        "Output clean plain text only."
            }
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert OCR and document text extraction assistant. "
                        "Extract accurate, clean plain text from document images. "
                        "Maintain logical reading order and structure clues (headings, bullet points, etc.) "
                        "but do not add markdown unless absolutely necessary for clarity. "
                        "Do not invent or hallucinate content."
                    )},
                    {"role": "user", "content": single_message}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            
            page_text = response.choices[0].message.content.strip()
            result += page_text + "\n\n"  # Separate pages with blank lines
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing page {i+1}: {str(e)}")

    return result.strip()


@app.post("/app-extract-text", response_model=ExtractionResponse)
async def extract_text_endpoint(file: UploadFile = File(...)):
    """
    FastAPI endpoint to upload a PDF and extract plain text from it.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid PDF.")

    try:
        extracted_text = await app_extract_text_from_pdf(file)
        
        return ExtractionResponse(
            extracted_text=extracted_text,
            page_count=extracted_text.count('\n\n') + 1  # rough estimate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")


@app.post("/v1/indic-summarize-pdf-all",
          response_model=IndicSummarizeAllPDFResponse,
          summary="Summarize and Translate a Specific Page of a PDF",
          description="Summarize the content of a specific page of a PDF file and translate the summary into the target language using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, summary, and translated summary", "model": IndicSummarizeAllPDFResponse},
              400: {"description": "Invalid PDF, page number, or language codes"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def indic_summarize_pdf_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),
    model: str = Form(default="gemma4", description="LLM model", enum=["gemma4"])  # Adjust SUPPORTED_MODELS as needed
):
    logger.debug(f"Processing indic summarize PDF: model={model}, tgt_lang={tgt_lang}, file={file.filename}")

    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Validate inputs (assuming validate_model and validate_language are defined)
        validate_model(model)
        validate_language(tgt_lang, "target language")

        #text_response = await extract_text_from_pdf(file, model)
        text_response_string = await extract_text_file(file)
        client = get_openai_client(model)

        # Single API call with structured JSON output
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a multilingual summarizer. "
                            "Always respond with valid JSON only, using exactly these keys:\n"
                            "- 'original_summary': summary in the original language of the text (3-5 sentences)\n"
                            "- 'translated_summary': summary in the target language specified below (3-5 sentences)\n"
                            "Do not add any extra text or explanations."
                },
                {
                    "role": "user",
                    "content": f"Target language for translation: {tgt_lang}\n\n"
                            f"First, detect the language of the following text.\n"
                            f"Then, summarize it in 3-5 sentences in its original language.\n"
                            f"Finally, provide the same summary in {tgt_lang}.\n\n"
                            f"Return only a JSON object with 'original_summary' and 'translated_summary'.\n\n"
                            f"Text:\n{text_response_string}"
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}  # Strongly enforces JSON output
        )

        raw_content = response.choices[0].message.content.strip()

        try:
            import json
            summaries = json.loads(raw_content)
            original_summary = summaries.get("original_summary", "").strip()
            translated_summary = summaries.get("translated_summary", "").strip()
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse model response as JSON")

        if not original_summary or not translated_summary:
            raise HTTPException(status_code=500, detail="One or both summaries are missing")

        return JSONResponse(content={
            "original_text": text_response_string,
            "summary": original_summary,              # In the source language (auto-detected)
            "translated_summary": translated_summary  # In tgt_lang
        })
    except Exception as e:
        logger.error("External indic custom prompt PDF API error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    

# Custom Prompt PDF Endpoint
@app.post("/v1/custom-prompt-pdf",
          response_model=CustomPromptPDFResponse,
          summary="Process a PDF with a Custom Prompt",
          description="Extract text from a PDF page and process it with a custom prompt.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text and custom prompt response", "model": CustomPromptPDFResponse},
              400: {"description": "Invalid PDF, page number, or prompt"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    validate_model(model)

    logger.debug("Processing custom prompt PDF request", extra={
        "endpoint": "/v1/custom-prompt-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "prompt_length": len(prompt),
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/custom-prompt-pdf"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {"page_number": page_number, "prompt": prompt, "model": model}

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=30
        )
        response.raise_for_status()

        response_data = response.json()
        original_text = response_data.get("original_text", "")
        custom_response = response_data.get("response", "")
        processed_page = response_data.get("processed_page", page_number)

        if not original_text or not custom_response:
            logger.warning(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, response={'present' if custom_response else 'missing'}")
            return CustomPromptPDFResponse(
                original_text=original_text or "No text extracted",
                response=custom_response or "No response provided",
                processed_page=processed_page
            )

        logger.debug(f"Custom prompt PDF completed in {time.time() - start_time:.2f} seconds")
        return CustomPromptPDFResponse(
            original_text=original_text,
            response=custom_response,
            processed_page=processed_page
        )

    except requests.Timeout:
        logger.error("External custom prompt PDF API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External custom prompt PDF API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")

@app.post("/v1/indic-custom-prompt-pdf",
          response_model=IndicCustomPromptPDFResponse,
          summary="Process a PDF with a Custom Prompt and Translation",
          description="Extract text from a specific page of a PDF, process it with a custom prompt, and translate the response into a target language using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, custom prompt response, and translated response", "model": IndicCustomPromptPDFResponse},
              400: {"description": "Invalid PDF, page number, prompt, or language codes"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def indic_custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    query_lang: str = Form("eng_Latn", description="Query language code (e.g., eng_Latn)"),  # Default added
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),  # Default added
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Validate inputs
    validate_model(model)
    validate_language(query_lang, "query language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic custom prompt PDF request", extra={
        "endpoint": "/v1/indic-custom-prompt-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "prompt_length": len(prompt),
        "query_lang": query_lang,
        "tgt_lang": tgt_lang,
        "model": model,
        "client_ip": request.client.host
    })


    #external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-custom-prompt-pdf"
    start_time = time.time()

    try:

        # original_text = await extract_text(file)
        original_text = await extract_text_page(file, page_number)

        if not original_text.strip():
            raise HTTPException(status_code=500, detail="Extracted text is empty")

        client = get_openai_client(model)

        # First: Get the answer in the original language (or model's best language)
        answer_prompt = f"{prompt}:\n\n{original_text}"

        answer_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        query_answer = answer_response.choices[0].message.content.strip()

        # Second: Translate the answer to the target language
        translation_prompt = f"Translate the following text to {tgt_lang}. Return only the translated text, nothing else:\n\n{query_answer}"

        translation_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0.0,  # Low temperature for accurate translation
            max_tokens=500,
        )
        translated_query_answer = translation_response.choices[0].message.content.strip()

        processed_page = page_number


        logger.debug(f"Indic custom prompt PDF completed in {time.time() - start_time:.2f} seconds, page processed: {processed_page}")

        #processed_page = response_data.get("processed_page", page_number)

        if not original_text or not query_answer or not translated_query_answer:
            logger.warning(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, query_answer={'present' if query_answer else 'missing'}, translated_query_answer={'present' if translated_query_answer else 'missing'}")
            return IndicCustomPromptPDFResponse(
                original_text=original_text or "No text extracted",
                query_answer=query_answer or "No response provided",
                translated_query_answer=translated_query_answer or "No translated response provided",
                processed_page=processed_page
            )

        
        logger.debug(f"Indic custom prompt PDF completed in {time.time() - start_time:.2f} seconds, page processed: {processed_page}")
        return IndicCustomPromptPDFResponse(
            original_text=original_text,
            query_answer=query_answer,
            translated_query_answer=translated_query_answer,
            processed_page=processed_page
        )

    except requests.Timeout:
        logger.error("External indic custom prompt PDF API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External indic custom prompt PDF API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")



@app.post("/v1/indic-custom-prompt-pdf-all",
          response_model=IndicCustomPromptPDFAllResponse,
          summary="Process a PDF with a Custom Prompt and Translation",
          description="Extract text from a specific page of a PDF, process it with a custom prompt, and translate the response into a target language using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Extracted text, custom prompt response, and translated response", "model": IndicCustomPromptPDFAllResponse},
              400: {"description": "Invalid PDF, page number, prompt, or language codes"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def indic_custom_prompt_pdf_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    query_lang: str = Form("eng_Latn", description="Source language code (e.g., eng_Latn)"),  # Default added
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),  # Default added
    model: str = Form(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Validate inputs
    validate_model(model)
    validate_language(query_lang, "query language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic custom prompt PDF request", extra={
        "endpoint": "/v1/indic-custom-prompt-pdf-all",
        "file_name": file.filename,
        "prompt_length": len(prompt),
        "query_lang": query_lang,
        "tgt_lang": tgt_lang,
        "model": model,
        "client_ip": request.client.host
    })


    validate_language(tgt_lang, "target language")

    text_response = await extract_text_file(file)

    
    try:
    # Convert dictionary values to a single string
        text_response_string = text_response
        #text_response_string = "\n".join(str(value) for value in page_contents_dict.values() if value)
        
        if not text_response_string.strip():
            raise HTTPException(status_code=500, detail="Extracted text is empty")

        client = get_openai_client(model)

        system_prompt = f"Return the answer only in {tgt_lang} language"
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": f" {query_lang} :\n\n{text_response_string}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        summary = summary_response.choices[0].message.content
        
        if not summary:
            raise HTTPException(status_code=500, detail="Summary generation failed")

    
        original_text= text_response_string
        query_answer = summary
        translated_query_answer = summary
        if not original_text or not query_answer or not translated_query_answer:
            logger.warning(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, query_answer={'present' if query_answer else 'missing'}, translated_query_answer={'present' if translated_query_answer else 'missing'}")
            return IndicCustomPromptPDFAllResponse(
                original_text=original_text or "No text extracted",
                query_answer=query_answer or "No response provided",
                translated_query_answer=translated_query_answer or "No translated response provided",
                )

        #logger.debug(f"Indic custom prompt PDF completed in {time.time() - start_time:.2f} seconds")
        return IndicCustomPromptPDFAllResponse(
            original_text=original_text,
            query_answer=query_answer,
            translated_query_answer=translated_query_answer,
        )
    
    except requests.Timeout:
        logger.error("External indic custom prompt PDF API timed out")
    raise HTTPException(status_code=504, detail="External API timeout")


@app.post("/v1/indic-custom-prompt-kannada-pdf",
          summary="Generate Kannada PDF with Custom Prompt",
          description="Process a PDF with a custom prompt and generate a new PDF in Kannada using an external API.",
          tags=["PDF"],
          responses={
              200: {"description": "Generated Kannada PDF file", "content": {"application/pdf": {"example": "Binary PDF data"}}},
              400: {"description": "Invalid PDF, page number, prompt, or language"},
              500: {"description": "External API error"},
              504: {"description": "External API timeout"}
          })
async def indic_custom_prompt_kannada_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: int = Form(..., description="Page number to process (1-based indexing)"),
    prompt: str = Form(..., description="Custom prompt to process the page content (e.g., 'list key points')"),
    src_lang: str = Form(..., description="Source language code (e.g., eng_Latn)"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Validate page number
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    # Validate prompt
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")


    if src_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}. Must be one of {SUPPORTED_LANGUAGES}")

    logger.debug("Processing Kannada PDF generation request", extra={
        "endpoint": "/v1/indic-custom-prompt-kannada-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "prompt": prompt,
        "src_lang": src_lang,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-custom-prompt-kannada-pdf/"
    start_time = time.time()

    # Create a temporary file to store the generated PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file_path = temp_file.name

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
            "page_number": page_number,
            "prompt": prompt,
            "src_lang": src_lang
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        # Write the PDF content to the temporary file
        with open(temp_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Prepare headers for the response
        headers = {
            "Content-Disposition": "attachment; filename=\"generated_kannada.pdf\"",
            "Cache-Control": "no-cache",
        }

        # Schedule file cleanup as a background task
        def cleanup_file(file_path: str):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {file_path}: {str(e)}")

        background_tasks.add_task(cleanup_file, temp_file_path)

        logger.debug(f"Kannada PDF generation completed in {time.time() - start_time:.2f} seconds")
        return FileResponse(
            path=temp_file_path,
            filename="generated_kannada.pdf",
            media_type="application/pdf",
            headers=headers
        )

    except requests.Timeout:
        logger.error("External Kannada PDF API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External Kannada PDF API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    finally:
        # Close the temporary file to ensure it's fully written
        temp_file.close()


ocr_query_string = "Return the plain text extracted from this image."

def ocr_page_with_rolm_query(img_base64: str, query:str,  model: str) -> str:
    """Perform OCR on the provided base64 image using the specified model."""
    try:
        client = get_openai_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {"type": "text", "text": query}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


def vision_query(img_base64: str, user_query:str,  model: str, system_prompt:str) -> str:
    """Perform OCR on the provided base64 image using the specified model."""

    try:
        client = get_openai_client(model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt }]
                
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        },
                        {"type": "text", "text": user_query}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# Visual Query Endpoint
@app.post("/v1/ocr",
          response_model=OCRResponse,
          summary="OCR with Image",
          description="Process a Image as OCR",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": OCRResponse},
              400: {"description": "Invalid query, image"},
              422: {"description": "Validation error in request body"},
              504: {"description": "Visual query service timeout"}
          })
async def ocr_query(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    model: str = Query(default="gemma4", description="LLM model", enum=SUPPORTED_MODELS)
):
    # Validate model
    validate_model(model)

    logger.debug("Processing visual query direct request", extra={
        "endpoint": "/v1/ocr",
        "file_name": file.filename,
        "client_ip": request.client.host,
        "model": model
    })

    try:
        response = await ocr_image(file=file)
        
        answer = response.get("extracted_text", "")

        if not answer:
            logger.warning(f"Empty or missing 'response' field in external API response: {answer}")
            raise HTTPException(status_code=500, detail="No valid response provided by visual query direct service")

        logger.debug(f"Visual query direct successful: {answer}")
        return OCRResponse(answer=answer)

    except requests.Timeout:
        logger.error("Visual query direct request timed out")
        raise HTTPException(status_code=504, detail="Visual query direct service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query direct failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from visual query direct service")


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    try:
        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        text = ocr_page_with_rolm_query(img_base64, ocr_query_string ,  model="gemma4")
        return {"extracted_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


async def indic_visual_query_direct(
    file: UploadFile = File(..., description="PNG image file to analyze"),
    prompt: Optional[str] = Form(None, description="Optional custom prompt to process the extracted text"),
    model: str = Form("gemma4", description="LLM model", enum=["gemma4", "moondream", "smolvla"])
):
    try:
        if not file.content_type.startswith("image/png"):
            raise HTTPException(status_code=400, detail="Only PNG images supported")

        logger.debug(f"Processing indic visual query: model={model}, prompt={prompt[:50] if prompt else None}")

        image_bytes = await file.read()
        image = BytesIO(image_bytes)
        img_base64 = encode_image(image)
        extracted_text = ocr_page_with_rolm_query(img_base64,prompt, model)

        response = extracted_text

        result = {
            "extracted_text": extracted_text,
            "response": response
        }
        if response:
            result["response"] = response

        logger.debug(f"visual query direct successful: extracted_text_length={len(extracted_text)}")
        return JSONResponse(content=result)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error translating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error translating: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    external_api_base_url_llm = os.getenv("DWANI_API_BASE_URL_LLM")
    if not external_api_base_url_llm:
        raise ValueError("Environment variable DWANI_API_BASE_URL_LLM must be set")
        
    external_api_base_url_tts = os.getenv("DWANI_API_BASE_URL_TTS")
    if not external_api_base_url_tts:
        raise ValueError("Environment variable DWANI_API_BASE_URL_TTS must be set")
    
    external_api_base_url_speech_to_speech = os.getenv("DWANI_API_BASE_URL_S2S")
    if not external_api_base_url_speech_to_speech:
        raise ValueError("Environment variable DWANI_API_BASE_URL_S2S must be set")
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)