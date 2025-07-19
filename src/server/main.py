import argparse
import os
from typing import List
from abc import ABC, abstractmethod
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
import tempfile
import os
    
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Form, Query
from pydantic import BaseModel, Field

from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
import requests
from typing import List, Optional, Dict, Any


from time import time
from typing import Optional
# Assuming these are in your project structure
#from config.tts_config import SPEED, ResponseFormat, config as tts_config
#from config.logging_config import logger

import logging
import logging.config
from logging.handlers import RotatingFileHandler

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


# FastAPI app setup with enhanced docs
app = FastAPI(
    title="dwani.ai API",
    description="A multimodal Inference API desgined for Privacy",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
)

# Supported models
SUPPORTED_MODELS = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]

SUPPORTED_LANGUAGES = ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "eng_Latn", "deu_Latn"]

# Pydantic models (updated to include model validation)
class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

    class Config:
        schema_extra = {
            "example": {
                "query": "Describe the image",
                "src_lang": "eng_Latn",
                "tgt_lang": "kan_Knda",
                "model": "moondream"
            }
        }


class VisualQueryDirectRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

    class Config:
        schema_extra = {
            "example": {
                "query": "Describe the image",
                "model": "moondream"
            }
        }

class VisualQueryResponse(BaseModel):
    answer: str

    class Config:
        schema_extra = {"example": {"answer": "The image shows a screenshot of a webpage."}}


class VisualQueryDirectResponse(BaseModel):
    answer: str

    class Config:
        schema_extra = {"example": {"answer": "The image shows a screenshot of a webpage."}}

class PDFTextExtractionResponse(BaseModel):
    page_content: str = Field(..., description="Extracted text from the specified PDF page")

    class Config:
        schema_extra = {
            "example": {
                "page_content": "Google Interview Preparation Guide\nCustomer Engineer Specialist\n\nOur hiring process\n..."
            }
        }


class PDFTextExtractionAllResponse(BaseModel):
    page_contents: Dict[str, str] = Field(..., description="Extracted text from each PDF page, with page numbers as keys and text content as values")

    class Config:
        schema_extra = {
            "example": {
                "page_contents": {
                    "0": "Google Interview Preparation Guide\nCustomer Engineer Specialist\n\nOur hiring process\n...",
                    "1": "Page 2 content\nAdditional details about the interview process\n..."
                }
            }
        }


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

# Request/Response Models
class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

    class Config:
        schema_extra = {"example": {"text": "Hello, how are you?"}} 

class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

    class Config:
        schema_extra = {"example": {"text": "Hi there, I'm doing great!"}} 

class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

    class Config:
        schema_extra = {"example": {"result": "Processed audio output"}} 

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Hello, how are you?",
                "src_lang": "kan_Knda",
                "tgt_lang": "kan_Knda",
                "model": "gemma3"
            }
        }

class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 10000 characters)", max_length=10000)
    model: str = Field(default="gemma3", description="LLM model")
    system_prompt: str = Field(default="", description="System prompt")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Hello, how are you?",
                "model": "gemma3",
                "system_prompt": ""
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 


class ChatDirectResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

    class Config:
        schema_extra = {
            "example": {
                "sentences": ["Hello", "How are you?"],
                "src_lang": "en",
                "tgt_lang": "kan_Knda"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Text query")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model")
    class Config:
        schema_extra = {
            "example": {
                "query": "Describe the image",
                "src_lang": "kan_Knda",
                "tgt_lang": "kan_Knda",
                "model": "gemma3"
            }
        }

class VisualQueryResponse(BaseModel):
    answer: str

    class Config:
        schema_extra = {"example": {"answer": "The image shows a screenshot of a webpage."}}

# Endpoints with enhanced Swagger docs
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

from pathlib import Path
from openai import OpenAI

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
    allowed_languages = ["kannada", "hindi", "tamil", "english","german", "marathi", "telugu" ]
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
                timeout=90
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
    valid_models = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}, model: {chat_request.model}")

    try:
        # Construct the external URL based on the selected model
        base_url = os.getenv('DWANI_API_BASE_URL_LLM')
        external_url = f"{base_url}/indic_chat"

        payload = {
            "prompt": chat_request.prompt,
            "src_lang": chat_request.src_lang,
            "tgt_lang": chat_request.tgt_lang,
            "model": chat_request.model
        }

        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        response_text = response_data.get("response", "")
        logger.debug(f"Generated Chat response from external API: {response_text}, model: {chat_request.model}")

        return ChatResponse(response=response_text)

    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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
    valid_models = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
    if chat_request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from {valid_models}")

    logger.debug(f"Received prompt: {chat_request.prompt}, model: {chat_request.model}")

    try:
        # Construct the external URL based on the selected model
        base_url = os.getenv('DWANI_API_BASE_URL_LLM')
        external_url = f"{base_url}/chat_direct"

        payload = {
            "prompt": chat_request.prompt,
            "model": chat_request.model,
            "system_prompt" : chat_request.system_prompt
        }

        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        response_text = response_data.get("response", "")
        logger.debug(f"Generated Chat response from external API: {response_text}, model: {chat_request.model}")

        return ChatDirectResponse(response=response_text)

    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

import httpx
@app.post("/v1/transcribe/", 
          response_model=TranscriptionResponse,
          summary="Transcribe Audio File",
          description="Transcribe an audio file into text in the specified language.",
          tags=["Audio"],
          responses={
              200: {"description": "Transcription result", "model": TranscriptionResponse},
              400: {"description": "Invalid audio or language"},
              504: {"description": "Transcription service timeout"}
          })
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil, english, german)")
):
    # Validate language
    allowed_languages = ["kannada", "hindi", "tamil", "english","german" ]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    start_time = time.time()
   
    if( language in ["english", "german"]):
        
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type),
#                'model': (None, 'Systran/faster-whisper-large-v3')
                'model': (None, 'Systran/faster-whisper-small')
        }
        
        response = httpx.post('http://localhost:8000/v1/audio/transcriptions', files=files, timeout=90.0)

        if response.status_code == 200:
            transcription = response.json().get("text", "")
            if transcription:
                logger.debug(f"Transcription completed in {time.time() - start_time:.2f} seconds")
                return TranscriptionResponse(text=transcription)
            else:
                logger.debug("Transcription empty, try again.")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        else:
            logger.debug(f"Transcription error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    else: 
        try:
            file_content = await file.read()
            files = {"file": (file.filename, file_content, file.content_type)}
            
            external_url = f"{os.getenv('DWANI_API_BASE_URL_ASR')}/transcribe/?language={language}"
            response = requests.post(
                external_url,
                files=files,
                headers={"accept": "application/json"},
                timeout=90
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

@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of sentences from a source to a target language.",
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
    supported_languages = [
        "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
        "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
        "rus_Cyrl", "pol_Latn"
    ]
    if request.src_lang not in supported_languages or request.tgt_lang not in supported_languages:
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
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=90
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

class VisualQueryResponse(BaseModel):
    answer: str

    class Config:
        schema_extra = {"example": {"answer": "The image shows a screenshot of a webpage."}}

# Visual Query Endpoint
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
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    # Validate query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 10000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 10000 characters")

    # Validate language codes
    supported_languages = ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "eng_Latn", "deu_Latn"]
    if src_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}. Must be one of {supported_languages}")
    if tgt_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {tgt_lang}. Must be one of {supported_languages}")

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

    external_url = f"{os.getenv('DWANI_API_BASE_URL_VISION')}/indic-visual-query/"

    try:
        file_content = await file.read()
        if not file.content_type.startswith("image/png"):
            raise HTTPException(status_code=400, detail="Only PNG images supported")

        files = {"file": (file.filename, file_content, file.content_type)}
        data = {
            "prompt": query,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "model": model
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        answer = response_data.get("translated_response", "")

        if not answer:
            logger.warning(f"Empty or missing 'translated_response' field in external API response: {response_data}")
            raise HTTPException(status_code=500, detail="No valid response provided by visual query service")

        logger.debug(f"Visual query successful: {answer}")
        return VisualQueryResponse(answer=answer)

    except requests.Timeout:
        logger.error("Visual query request timed out")
        raise HTTPException(status_code=504, detail="Visual query service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from visual query service")



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
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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

    external_url = f"{os.getenv('DWANI_API_BASE_URL_VISION')}/visual-query-direct/"

    try:
        file_content = await file.read()
        if not file.content_type.startswith("image/png"):
            raise HTTPException(status_code=400, detail="Only PNG images supported")

        files = {"file": (file.filename, file_content, file.content_type)}
        data = {
            "prompt": query,
            "model": model
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        answer = response_data.get("response", "")

        if not answer:
            logger.warning(f"Empty or missing 'response' field in external API response: {response_data}")
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

from enum import Enum

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
            timeout=90
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
    

'''
Upgrading system to use Vllm server
'''

# Extract Text Endpoint
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
            timeout=90
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
            timeout=90

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
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    supported_languages = ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "eng_Latn"]
    if src_lang not in supported_languages or tgt_lang not in supported_languages:
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
            timeout=90
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
            timeout=90
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
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
            timeout=90
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

# Indic Summarize PDF Endpoint (Updated)
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
    tgt_lang: str = Form("kan_Knda", description="Target language code (e.g., kan_Knda)"),  # Default added
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    logger.debug(f"Processing indic summarize PDF: model={model}, tgt_lang={tgt_lang} and file={file.filename}")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Validate inputs
    validate_model(model)
    validate_language(tgt_lang, "target language")

    logger.debug("Processing Indic PDF summary request", extra={
        "endpoint": "/v1/indic-summarize-pdf-all",
        "file_name": file.filename,
        "tgt_lang": tgt_lang,
        "model": model,
        "client_ip": request.client.host
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-summarize-pdf-all"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
            "tgt_lang": tgt_lang,
            "model": model
        }

        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        original_text = response_data.get("original_text", "")
        summary = response_data.get("summary", "")
        translated_summary = response_data.get("translated_summary", "")

        if not original_text or not summary or not translated_summary:
            logger.debug(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, summary={'present' if summary else 'missing'}, translated_summary={'present' if translated_summary else 'missing'}")
            return IndicSummarizeAllPDFResponse(
                original_text=original_text or "No text extracted",
                summary=summary or "No summary provided",
                translated_summary=translated_summary or "No translated summary provided",
            )

        logger.debug(f"Indic PDF summary completed in {time.time() - start_time:.2f} seconds")
        return IndicSummarizeAllPDFResponse(
            original_text=original_text,
            summary=summary,
            translated_summary=translated_summary,
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
            timeout=90
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
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()

        original_text = response_data.get("original_text", "")
        query_answer = response_data.get("query_answer", "")
        translated_query_answer = response_data.get("translated_query_answer", "")

        processed_page = response_data.get("processed_page", page_number)

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
    model: str = Form(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
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

    external_url = f"{os.getenv('DWANI_API_BASE_URL_PDF')}/indic-custom-prompt-pdf-all"
    start_time = time.time()

    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, "application/pdf")}
        data = {
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
            timeout=90
        )
        response.raise_for_status()

        response_data = response.json()
        original_text = response_data.get("original_text", "")
        query_answer = response_data.get("query_answer", "")
        translated_query_answer = response_data.get("translated_query_answer", "")
        
        if not original_text or not query_answer or not translated_query_answer:
            logger.warning(f"Incomplete response from external API: original_text={'present' if original_text else 'missing'}, query_answer={'present' if query_answer else 'missing'}, translated_query_answer={'present' if translated_query_answer else 'missing'}")
            return IndicCustomPromptPDFAllResponse(
                original_text=original_text or "No text extracted",
                query_answer=query_answer or "No response provided",
                translated_query_answer=translated_query_answer or "No translated response provided",
                )

        logger.debug(f"Indic custom prompt PDF completed in {time.time() - start_time:.2f} seconds")
        return IndicCustomPromptPDFAllResponse(
            original_text=original_text,
            query_answer=query_answer,
            translated_query_answer=translated_query_answer,
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

    # Validate source language
    supported_languages = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu"]
    if src_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}. Must be one of {supported_languages}")

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
            timeout=90
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


from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

import time

# vLLM server configuration
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:9000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

# HTTP headers for vLLM requests
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {VLLM_API_KEY}" if VLLM_API_KEY else None
}
headers = {k: v for k, v in headers.items() if v is not None}

# In-memory storage for rate limiting
rate_limit_store = defaultdict(list)

# Pydantic models for OpenAI-compatible API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None

# Custom features
class ProxyFeatures:
    @staticmethod
    def log_request(request: Dict, client_ip: str) -> None:
        """Log incoming request details."""
        logger.debug(
            f"Request from {client_ip}: model={request.get('model')}, "
            f"messages={len(request.get('messages', []))} messages"
        )

    @staticmethod
    def log_response(response: Dict, processing_time: Optional[float] = None) -> None:
        """Log response details."""
        logger.debug(
            f"Response: id={response.get('id')}, choices={len(response.get('choices', []))}, "
            f"processing_time={processing_time:.2f}s" if processing_time else "processing_time=unknown"
        )

    @staticmethod
    def rate_limit(client_ip: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
        """In-memory rate limiting per client IP."""
        current_time = time.time()
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip] if current_time - t < window_seconds
        ]
        rate_limit_store[client_ip].append(current_time)
        if len(rate_limit_store[client_ip]) > max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
        return True

    @staticmethod
    def modify_response(response: Dict) -> Dict:
        """Modify response (e.g., add disclaimer)."""
        for choice in response.get('choices', []):
            choice['message']['content'] = (
                f"{choice['message']['content']}\n\n*Disclaimer: Generated by AI, verify before use.*"
            )
        return response

    @staticmethod
    def estimate_usage(request: Dict, response: Dict) -> Dict:
        """Estimate token usage if not provided by vLLM."""
        if response.get('usage') is None or not all(
            key in response['usage'] for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']
        ):
            prompt_text = ' '.join(msg.get('content', '') for msg in request.get('messages', []))
            response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            # Rough token estimation (1 token ~ 4 characters, minimum 1 token)
            prompt_tokens = max(len(prompt_text) // 4, 1)
            completion_tokens = max(len(response_text) // 4, 1)
            response['usage'] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
        return response

from pydantic import BaseModel, ValidationError

# Chat completions endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, fastapi_request: Request):
    """Proxy chat completion requests to vLLM with custom features."""
    start_time = time.time()
    client_ip = fastapi_request.client.host

    # Log request
    ProxyFeatures.log_request(request.dict(), client_ip)

    # Rate limiting
    if not ProxyFeatures.rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Forward request to vLLM
    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not supported yet")

    try:
        response = requests.post(
            f"{VLLM_API_BASE}/chat/completions",
            json=request.dict(),
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        vllm_response = response.json()

        # Log raw response for debugging
        logger.debug(f"vLLM raw response: {vllm_response}")

        # Estimate usage if missing
        vllm_response = ProxyFeatures.estimate_usage(request.dict(), vllm_response)

        # Modify response
        modified_response = ProxyFeatures.modify_response(vllm_response)

        # Log response
        processing_time = time.time() - start_time
        ProxyFeatures.log_response(modified_response, processing_time)

        # Validate and return response
        return ChatCompletionResponse(**modified_response)

    except requests.RequestException as e:
        logger.error(f"vLLM server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"vLLM error: {str(e)}")
    except ValidationError as e:
        logger.error(f"Response validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Response validation error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}




if __name__ == "__main__":
    # Ensure EXTERNAL_API_BASE_URL is set
    external_api_base_url_pdf = os.getenv("DWANI_API_BASE_URL_PDF")
    if not external_api_base_url_pdf:
        raise ValueError("Environment variable DWANI_API_BASE_URL_PDF must be set")
    
    external_api_base_url_vision = os.getenv("DWANI_API_BASE_URL_VISION")
    if not external_api_base_url_vision:
        raise ValueError("Environment variable DWANI_API_BASE_URL_VISION must be set")
    
    external_api_base_url_llm = os.getenv("DWANI_API_BASE_URL_LLM")
    if not external_api_base_url_llm:
        raise ValueError("Environment variable DWANI_API_BASE_URL_LLM must be set")
    
    external_api_base_url_llm_qwen3 = os.getenv("DWANI_API_BASE_URL_LLM_QWEN")
    if not external_api_base_url_llm_qwen3:
        raise ValueError("Environment variable DWANI_API_BASE_URL_LLM_QWEN must be set")
    
    external_api_base_url_tts = os.getenv("DWANI_API_BASE_URL_TTS")
    if not external_api_base_url_tts:
        raise ValueError("Environment variable DWANI_API_BASE_URL_TTS must be set")
    
    external_api_base_url_asr = os.getenv("DWANI_API_BASE_URL_ASR")
    if not external_api_base_url_asr:
        raise ValueError("Environment variable DWANI_API_BASE_URL_ASR must be set")
    
    external_api_base_url_translate = os.getenv("DWANI_API_BASE_URL_TRANSLATE")
    if not external_api_base_url_translate:
        raise ValueError("Environment variable DWANI_API_BASE_URL_TRANSLATE must be set")
    
    external_api_base_url_speech_to_speech = os.getenv("DWANI_API_BASE_URL_S2S")
    if not external_api_base_url_speech_to_speech:
        raise ValueError("Environment variable DWANI_API_BASE_URL_S2S must be set")
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)