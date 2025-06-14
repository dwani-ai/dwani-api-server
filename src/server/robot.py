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

from openai import AsyncOpenAI, OpenAIError

from time import time
from typing import Optional
# Assuming these are in your project structure
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from config.logging_config import logger

# FastAPI app setup with enhanced docs
app = FastAPI(
    title="Dhwani API",
    description="A multilingual AI-powered API supporting Indian languages for chat, text-to-speech, audio processing, and transcription.",
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
        "https://dwani-*.hf.space"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

    class Config:
        schema_extra = {"example": {"text": "Hi there, I'm doing great!"}} 


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 1000 characters)", max_length=1000)
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
    prompt: str = Field(..., description="Prompt for chat (max 1000 characters)", max_length=1000)
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
    if len(chat_request.prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 1000 characters")

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
            timeout=60
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
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 1000 characters")

    # Validate language codes
    supported_languages = ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "eng_Latn", "deu_Latn"]
    if src_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}. Must be one of {supported_languages}")
    if tgt_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {tgt_lang}. Must be one of {supported_languages}")

    # Validate model
    validate_model(model)

    logger.info("Processing visual query request", extra={
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
            timeout=60
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
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 1000 characters")

 
    # Validate model
    validate_model(model)

    logger.info("Processing visual query direct request", extra={
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
            timeout=60
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




# Visual Query Endpoint
@app.post("/v1/visual_query_raw",
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
async def visual_query_direct_raw(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image (e.g., 'describe the image')"),
    model: str = Query(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
):
    # Validate query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

 
    # Validate model
    validate_model(model)

    logger.info("Processing visual query direct request", extra={
        "endpoint": "/v1/visual_query_direct",
        "query_length": len(query),
        "client_ip": request.client.host,
        "model": model
    })

    external_url = f"{os.getenv('DWANI_API_BASE_URL_VISION')}/visual-query-raw/"

    try:
        data = {
            "prompt": query,
            "model": model
        }

        response = requests.post(
            external_url,
            data=data,
            headers={"accept": "application/json"},
            timeout=60
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

    

if __name__ == "__main__":
    # Ensure EXTERNAL_API_BASE_URL is set
    
    external_api_base_url_vision = os.getenv("DWANI_API_BASE_URL_VISION")
    if not external_api_base_url_vision:
        raise ValueError("Environment variable DWANI_API_BASE_URL_VISION must be set")
    
    external_api_base_url_llm = os.getenv("DWANI_API_BASE_URL_LLM")
    if not external_api_base_url_llm:
        raise ValueError("Environment variable DWANI_API_BASE_URL_LLM must be set")
    
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)