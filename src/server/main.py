import argparse
import os
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Depends, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse, RedirectResponse
from fastapi.background import BackgroundTasks
import tempfile
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
import requests
import httpx
import json
from time import time
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from num2words import num2words
from datetime import datetime
import pytz
from pdf2image import convert_from_path
from io import BytesIO
import base64
import re
import cv2
import numpy as np

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

app = FastAPI(
    title="dwani.ai API",
    description="A multimodal Inference API designed for Privacy",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Utility", "description": "General utility endpoints"},
        {"name": "PDF", "description": "PDF processing endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.hf.space",
        "https://dwani.ai",
        "https://*.dwani.ai",
        "https://dwani-*.hf.space",
        "http://localhost:11080",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_MODELS = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
SUPPORTED_LANGUAGES = [
    "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "guj_Gujr", "mni_Mtei",
    "npi_Deva", "urd_Arab", "ory_Orya",
    "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
    "rus_Cyrl", "pol_Latn",
]

language_options = [
    ("English", "eng_Latn"), ("Kannada", "kan_Knda"), ("Hindi", "hin_Deva"),
    ("Assamese", "asm_Beng"), ("Bengali", "ben_Beng"), ("Gujarati", "guj_Gujr"),
    ("Malayalam", "mal_Mlym"), ("Marathi", "mar_Deva"), ("Odia", "ory_Orya"),
    ("Punjabi", "pan_Guru"), ("Tamil", "tam_Taml"), ("Telugu", "tel_Telu"),
    ("German", "deu_Latn"),
]

def get_language_name(lang_code):
    for name, code in language_options:
        if code == lang_code:
            return name
    return "English"

def validate_model(model: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}. Must be one of {SUPPORTED_MODELS}")
    return model

def validate_language(lang: str, field_name: str) -> str:
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {lang}. Must be one of {SUPPORTED_LANGUAGES}")
    return lang

def get_openai_client(model: str) -> OpenAI:
    valid_models = ["gemma3", "moondream", "qwen3", "sarvam-m", "gpt-oss"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")
    model_ports = {
        "qwen3": "9100",
        "gemma3": "9000",
        "moondream": "7882",
        "gpt-oss": "9500",
        "sarvam-m": "7884",
    }
    base_url = f"http://0.0.0.0:{model_ports[model]}/v1"
    return OpenAI(api_key="http", base_url=base_url)

def encode_image(image: BytesIO) -> str:
    return base64.b64encode(image.read()).decode("utf-8")

async def render_pdf_to_png(pdf_file):
    temp_file_path = "temp.pdf"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(await pdf_file.read())
        images = convert_from_path(temp_file_path, fmt="jpeg")
        if not images:
            logger.error("No images generated from PDF")
            raise HTTPException(status_code=400, detail="Failed to convert PDF to images: No pages found")
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to convert PDF to images: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {str(e)}")

async def get_base64_msg_from_pdf(file):
    try:
        images = await render_pdf_to_png(file)
    except Exception as e:
        logger.error(f"Failed to render PDF to PNG: {str(e)}")
        return []

    messages = []
    for i, image in enumerate(images):
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_bytes_io = BytesIO()
            image.save(image_bytes_io, format="JPEG", quality=85)
            image_bytes_io.seek(0)
            image_bytes = image_bytes_io.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            try:
                base64.b64decode(image_base64, validate=True)
            except Exception as e:
                logger.error(f"Invalid base64 string for page {i}: {str(e)}")
                continue
            messages.append({
                "type": "image_url",
                "image_url": {"url": image_base64}  # Raw base64 string
            })
        except Exception as e:
            logger.error(f"Image processing failed for page {i}: {str(e)}")
            continue
    return messages

def sanitize_json_string(s: str) -> str:
    if not s:
        return "{}"
    s = re.sub(r'[\x00-\x1F\x7F]', lambda m: '\\u{:04x}'.format(ord(m.group())), s)
    s = re.sub(r'[\n\t]+(?=[\{\[\]\},:0-9])', ' ', s)
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    s = s.strip()
    if not s.startswith('{') and not s.startswith('['):
        s = '{' + s + '}'
    return s

# Pydantic models
class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class OCRRequest(BaseModel):
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class VisualQueryDirectRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class VisualQueryResponse(BaseModel):
    answer: str

class OCRResponse(BaseModel):
    answer: str

class VisualQueryDirectResponse(BaseModel):
    answer: str

class PDFTextExtractionResponse(BaseModel):
    page_content: str = Field(..., description="Extracted text from the specified PDF page")

class PDFTextExtractionAllResponse(BaseModel):
    page_contents: Dict[str, str] = Field(..., description="Extracted text from each PDF page")

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
    original_text: str = Field(..., description="Extracted text from all pages")
    summary: str = Field(..., description="Summary of all pages in the source language")
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
    original_text: str = Field(..., description="Extracted text from all pages")
    query_answer: str = Field(..., description="Response based on the custom prompt")
    translated_query_answer: str = Field(..., description="Translated response in the target language")

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model")

class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    model: str = Field(default="gemma3", description="LLM model")
    system_prompt: str = Field(default="", description="System prompt")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

class ChatDirectResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

# Endpoints
@app.get("/v1/health", summary="Check API Health", description="Returns the health status of the API and the current model in use.", tags=["Utility"])
async def health_check():
    return {"status": "healthy", "model": "llm_model_name"}

@app.get("/", summary="Redirect to Docs", description="Redirects to the Swagger UI documentation.", tags=["Utility"])
async def home():
    return RedirectResponse(url="/docs")

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

@app.post("/v1/audio/speech", summary="Generate Speech from Text", description="Convert text to speech using an external TTS service.", tags=["Audio"])
async def generate_audio(
    request: Request,
    input: str = Query(..., description="Text to convert to speech (max 10000 characters)"),
    response_format: str = Query("mp3", description="Audio format (ignored, defaults to mp3)"),
    language: str = Query("kannada", description="Language for TTS"),
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

    allowed_languages = ["kannada", "hindi", "tamil", "english", "german", "telugu", "marathi"]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    start_time = time.time()
   
    if language in ["english", "german"]:
        openai = OpenAI(base_url="http://localhost:8000/v1", api_key="cant-be-empty")
        model_id = "speaches-ai/Kokoro-82M-v1.0-ONNX"
        voice_id = "af_heart"
        res = openai.audio.speech.create(
            model=model_id,
            voice=voice_id,
            input=input,
            response_format="wav",
            speed=1,
        )
        headers = {
            "Content-Disposition": "attachment; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
        }
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
            with open(temp_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            headers = {
                "Content-Disposition": "attachment; filename=\"speech.mp3\"",
                "Cache-Control": "no-cache",
            }
            def cleanup_file(file_path: str):
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                        logger.debug(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary file {file_path}: {str(e)}")
            background_tasks.add_task(cleanup_file, temp_file_path)
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
            temp_file.close()

@app.post("/v1/indic_chat", response_model=ChatResponse, summary="Chat with AI", description="Generate a chat response with translation support.", tags=["Chat"])
async def chat_v2(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 10000 characters")

    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}, model: {chat_request.model}")

    valid_models = ["gemma3", "qwen3", "sarvam-m", "gpt-oss"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    settings = get_settings()
    language_name = get_language_name(chat_request.tgt_lang)
    system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Do not explain. Return answer only in {language_name}"

    try:
        prompt_to_process = chat_request.prompt
        client = get_openai_client(chat_request.model)
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

@app.post("/v1/chat_direct", response_model=ChatDirectResponse, summary="Chat with AI", description="Generate a chat response from a prompt.", tags=["Chat"])
async def chat_direct(request: Request, chat_request: ChatDirectRequest):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 10000 characters")

    valid_models = ["gemma3", "qwen3", "sarvam-m", "gpt-oss"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    settings = get_settings()
    logger.debug(f"Received prompt: {chat_request.prompt}, model: {chat_request.model}")

    try:
        prompt_to_process = chat_request.prompt
        system_prompt = chat_request.system_prompt or f"You are Dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {time_to_words()}"
        client = get_openai_client(chat_request.model)
        response = client.chat.completions.create(
            model=chat_request.model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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

from enum import Enum

class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

@app.post("/v1/transcribe/", response_model=TranscriptionResponse, summary="Transcribe Audio File", description="Transcribe an audio file into text.", tags=["Audio"])
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil, english, german, telugu, marathi)")
):
    allowed_languages = ["kannada", "hindi", "tamil", "english", "german", "telugu", "marathi"]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    start_time = time.time()
    if language in ["english", "german"]:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type), 'model': (None, 'Systran/faster-whisper-small')}
        response = httpx.post('http://localhost:8000/v1/audio/transcriptions', files=files, timeout=30.0)
        if response.status_code == 200:
            transcription = response.json().get("text", "")
            if transcription:
                logger.debug(f"Transcription completed in {time.time() - start_time:.2f} seconds")
                return TranscriptionResponse(text=transcription)
            else:
                logger.debug("Transcription empty, try again.")
                raise HTTPException(status_code=500, detail="Transcription failed: Empty response")
        else:
            logger.debug(f"Transcription error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {response.text}")
    else: 
        try:
            file_content = await file.read()
            files = {"file": (file.filename, file_content, file.content_type)}
            external_url = f"{os.getenv('DWANI_API_BASE_URL_ASR')}/transcribe/?language={language}"
            response = requests.post(
                external_url,
                files=files,
                headers={"accept": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            transcription = response.json().get("text", "")
            logger.debug(f"Transcription completed in {time.time() - start_time:.2f} seconds")
            return TranscriptionResponse(text=transcription)
        except requests.Timeout:
            logger.error("Transcription service timed out")
            raise HTTPException(status_code=504, detail="Transcription service timeout")
        except requests.RequestException as e:
            logger.error(f"Transcription request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/translate", response_model=TranslationResponse, summary="Translate Text", description="Translate a list of sentences.", tags=["Translation"])
async def translate(request: TranslationRequest):
    if not request.sentences:
        raise HTTPException(status_code=400, detail="Sentences cannot be empty")
    
    if request.src_lang not in SUPPORTED_LANGUAGES or request.tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language codes: src={request.src_lang}, tgt={request.tgt_lang}")

    logger.debug(f"Received translation request: {len(request.sentences)} sentences, src_lang: {request.src_lang}, tgt_lang: {request.tgt_lang}")

    external_url = f"{os.getenv('DWANI_API_BASE_URL_TRANSLATE')}"
    payload = {
        "sentences": request.sentences,
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang
    }
    try:
        response = requests.post(
            f"{external_url}/translate?src_lang={request.src_lang}&tgt_lang={request.tgt_lang}",
            json=payload,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        translations = response_data.get("translations", [])
        if not translations or len(translations) != len(request.sentences):
            logger.warning(f"Unexpected response format: {response_data}")
            raise HTTPException(status_code=500, detail="Invalid response from translation service")
        logger.debug(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except requests.Timeout:
        logger.error("Translation request timed out")
        raise HTTPException(status_code=504, detail="Translation service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from translation service")

@app.post("/v1/indic_visual_query", response_model=VisualQueryResponse, summary="Visual Query with Image", description="Process a visual query with image and translation.", tags=["Chat"])
async def visual_query(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    src_lang: str = Query(..., description="Source language code"),
    tgt_lang: str = Query(..., description="Target language code"),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 10000 characters")
    if src_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}")
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {tgt_lang}")
    validate_model(model)
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    logger.debug("Processing visual query request", extra={
        "endpoint": "/v1/indic_visual_query",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": request.client.host,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model
    })

    image_bytes = await file.read()
    image = BytesIO(image_bytes)
    img_base64 = encode_image(image)
    language_name = get_language_name(tgt_lang)
    system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Return answer only in {language_name}"
    extracted_text = vision_query(img_base64, query, model, system_prompt=system_prompt)
    logger.debug(f"Visual query successful: extracted_text_length={len(extracted_text)}")
    return VisualQueryResponse(answer=extracted_text)

@app.post("/v1/visual_query_direct", response_model=VisualQueryDirectResponse, summary="Visual Query with Image", description="Process a visual query with image.", tags=["Chat"])
async def visual_query_direct(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 10000 characters")
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
        if isinstance(response, JSONResponse):
            response_body = json.loads(response.body.decode("utf-8"))
            answer = response_body.get("response", "")
        else:
            answer = response.get("response", "")
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

@app.post("/v1/speech_to_speech", summary="Speech-to-Speech Conversion", description="Convert input speech to processed speech.", tags=["Audio"])
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil)")
) -> StreamingResponse:
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

@app.post("/v1/extract-text", response_model=PDFTextExtractionResponse, summary="Extract Text from PDF", description="Extract text from a specified page of a PDF.", tags=["PDF"])
async def extract_text(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    page_number: int = Query(1, description="Page number to extract text from (1-based indexing)", ge=1),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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

@app.post("/v1/extract-text-all", response_model=PDFTextExtractionAllResponse, summary="Extract Text from PDF", description="Extract text from all pages of a PDF.", tags=["PDF"])
async def extract_text_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
        try:
            validated_response = PDFTextExtractionAllResponse(**response_data)
            extracted_text = validated_response.page_contents
        except Exception as e:
            logger.warning(f"Failed to validate response with Pydantic model: {str(e)}")
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

@app.post("/v1/extract-text-all-chunk", response_model=PDFTextExtractionAllResponse, summary="Extract Text from PDF", description="Extract text from all pages of a PDF.", tags=["PDF"])
async def extract_text_all_chunk(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
        try:
            validated_response = PDFTextExtractionAllResponse(**response_data)
            extracted_text = validated_response.page_contents
        except Exception as e:
            logger.warning(f"Failed to validate response with Pydantic model: {str(e)}")
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

@app.post("/v1/indic-extract-text/", response_model=DocumentProcessResponse, summary="Extract and Translate Text from PDF", description="Extract and translate text from a PDF page.", tags=["PDF"])
async def extract_and_translate(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    src_lang: str = Form("eng_Latn", description="Source language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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

@app.post("/v1/summarize-pdf", response_model=SummarizePDFResponse, summary="Summarize a Specific Page of a PDF", description="Summarize a specific page of a PDF.", tags=["PDF"])
async def summarize_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    page_number: int = Form(..., description="Page number to summarize (1-based indexing)", ge=1),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
            logger.warning(f"Incomplete response: original_text={'present' if original_text else 'missing'}, summary={'present' if summary else 'missing'}")
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

@app.post("/v1/indic-summarize-pdf", response_model=IndicSummarizePDFResponse, summary="Summarize and Translate a Specific Page of a PDF", description="Summarize and translate a specific page of a PDF.", tags=["PDF"])
async def indic_summarize_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    page_number: int = Form(..., description="Page number to summarize (1-based indexing)", ge=1),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    logger.debug(f"Processing indic summarize PDF: page_number={page_number}, model={model}, tgt_lang={tgt_lang}, file={file.filename}")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
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
            logger.debug(f"Incomplete response: original_text={'present' if original_text else 'missing'}, summary={'present' if summary else 'missing'}, translated_summary={'present' if translated_summary else 'missing'}")
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

async def extract_text_from_pdf(file: UploadFile = File(...), model: str = Body("gemma3", embed=True)) -> JSONResponse:
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported.")
        validate_model(model)
        ocr_query_string = "Return the plain text extracted from this image."
        pages = await get_base64_msg_from_pdf(file)
        if not pages:
            logger.warning("No pages extracted from PDF")
            raise HTTPException(status_code=400, detail="No pages extracted from PDF")
        page_contents = {}
        client = get_openai_client(model)
        for page_num, message in enumerate(pages):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                message,
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
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")
                page_contents[str(page_num)] = ""
                continue
        if not any(page_contents.values()):
            logger.error("No text extracted from any page")
            raise HTTPException(status_code=400, detail="No text extracted from PDF pages")
        return JSONResponse(content={"page_contents": page_contents})
    except Exception as e:
        logger.error(f"Error in extract_text_from_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        await file.close()

async def extract_text_batch_from_pdf(
    file: UploadFile = File(...),
    model: str = Body("gemma3", embed=True)
) -> JSONResponse:
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
                max_tokens=50000
            )
            raw_response = response.choices[0].message.content
            logger.debug("Raw OCR response length: %d, content: %s", len(raw_response), raw_response[:500])
            cleaned_response = re.sub(r'^```(?:json)?\n|\n```$', '', raw_response, flags=re.MULTILINE).strip()
            logger.debug("Cleaned response before sanitization: %s", cleaned_response[:500])
            cleaned_response = sanitize_json_string(cleaned_response)
            logger.debug("Sanitized response: %s", cleaned_response[:500])
            try:
                page_contents = json.loads(cleaned_response)
                logger.debug("Parsed page contents: %s", page_contents)
                if not isinstance(page_contents, dict):
                    raise ValueError("Response is not a valid JSON object with page numbers as keys")
            except json.JSONDecodeError as e:
                logger.error("JSON parsing failed: %s. Attempting fallback parsing.", str(e))
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

@app.post("/v1/indic-summarize-pdf-all", response_model=IndicSummarizeAllPDFResponse, summary="Summarize and Translate All Pages of a PDF", description="Summarize and translate all pages of a PDF.", tags=["PDF"])
async def indic_summarize_pdf_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to summarize"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=["gemma3"])
):
    logger.debug(f"Processing indic summarize PDF: model={model}, tgt_lang={tgt_lang}, file={file.filename}")
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        validate_model(model)
        validate_language(tgt_lang, "target language")
        text_response = await extract_text_from_pdf(file, model)
        try:
            page_contents_dict = json.loads(text_response.body.decode())["page_contents"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse text_response: %s", str(e))
            raise HTTPException(status_code=500, detail="Invalid OCR response format")
        if not page_contents_dict:
            logger.error("No pages extracted from PDF")
            raise HTTPException(status_code=400, detail="No pages extracted from PDF")
        text_response_string = "\n".join(str(value) for value in page_contents_dict.values() if value)
        if not text_response_string.strip():
            logger.error("Extracted text is empty")
            raise HTTPException(status_code=400, detail="Extracted text is empty")
        client = get_openai_client(model)
        language_name = get_language_name(tgt_lang)
        system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Return answer only in {language_name}"
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": f"Summarize the following text in 3-5 sentences:\n\n{text_response_string}"}]}
            ],
            temperature=0.3,
            max_tokens=500
        )
        summary = summary_response.choices[0].message.content
        if not summary:
            logger.error("Summary generation failed")
            raise HTTPException(status_code=500, detail="Summary generation failed")
        return IndicSummarizeAllPDFResponse(
            original_text=text_response_string,
            summary=summary,
            translated_summary=summary
        )
    except requests.Timeout:
        logger.error("External indic custom prompt PDF API timed out")
        raise HTTPException(status_code=504, detail="External API timeout")
    except Exception as e:
        logger.error(f"External indic custom prompt PDF API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")

@app.post("/v1/custom-prompt-pdf", response_model=CustomPromptPDFResponse, summary="Process a PDF with a Custom Prompt", description="Process a PDF page with a custom prompt.", tags=["PDF"])
async def custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
            logger.warning(f"Incomplete response: original_text={'present' if original_text else 'missing'}, response={'present' if custom_response else 'missing'}")
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

@app.post("/v1/indic-custom-prompt-pdf", response_model=IndicCustomPromptPDFResponse, summary="Process a PDF with a Custom Prompt and Translation", description="Process a PDF page with a custom prompt and translate.", tags=["PDF"])
async def indic_custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    query_lang: str = Form("eng_Latn", description="Query language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
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
    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-custom-prompt-pdf"
    start_time = time.time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
            "page_number": str(page_number),
            "prompt": prompt,
            "query_language": query_lang,
            "target_language": tgt_lang,
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
        query_answer = response_data.get("query_answer", "")
        translated_query_answer = response_data.get("translated_query_answer", "")
        processed_page = response_data.get("processed_page", page_number)
        if not original_text or not query_answer or not translated_query_answer:
            logger.warning(f"Incomplete response: original_text={'present' if original_text else 'missing'}, query_answer={'present' if query_answer else 'missing'}, translated_query_answer={'present' if translated_query_answer else 'missing'}")
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

@app.post("/v1/indic-custom-prompt-pdf-all", response_model=IndicCustomPromptPDFAllResponse, summary="Process a PDF with a Custom Prompt and Translation", description="Process all pages of a PDF with a custom prompt and translate.", tags=["PDF"])
async def indic_custom_prompt_pdf_all(
    request: Request,
    file: UploadFile = File(..., description="PDF file to process"),
    prompt: str = Form(..., description="Custom prompt to process the page content"),
    query_lang: str = Form("eng_Latn", description="Source language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
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
    try:
        text_response = await extract_text_from_pdf(file, model)
        try:
            page_contents_dict = json.loads(text_response.body.decode())["page_contents"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse text_response: %s", str(e))
            raise HTTPException(status_code=500, detail="Invalid OCR response format")
        if not page_contents_dict:
            logger.error("No pages extracted from PDF")
            raise HTTPException(status_code=400, detail="No pages extracted from PDF")
        text_response_string = "\n".join(str(value) for value in page_contents_dict.values() if value)
        if not text_response_string.strip():
            logger.error("Extracted text is empty")
            raise HTTPException(status_code=400, detail="Extracted text is empty")
        client = get_openai_client(model)
        language_name = get_language_name(tgt_lang)
        system_prompt = f"You are dwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. Return answer only in {language_name}"
        summary_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": f"{prompt}:\n\n{text_response_string}"}]}
            ],
            temperature=0.3,
            max_tokens=500
        )
        query_answer = summary_response.choices[0].message.content
        if not query_answer:
            logger.error("Query response generation failed")
            raise HTTPException(status_code=500, detail="Query response generation failed")
        return IndicCustomPromptPDFAllResponse(
            original_text=text_response_string,
            query_answer=query_answer,
            translated_query_answer=query_answer
        )
    except requests.Timeout:
        logger.error