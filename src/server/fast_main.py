
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import asyncio

from io import BytesIO
from PIL import Image
import base64



import logging
import logging.config
from logging.handlers import RotatingFileHandler


import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks


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
    model = "gemma3"  # or whatever vision model you're using (e.g., gpt-4o, gemma3, etc.)
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


import base64
from io import BytesIO
from pdf2image import convert_from_path
import os
import asyncio
import re

async def render_pdf_to_png(pdf_file):
    """Convert PDF to images."""
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

from openai import AsyncOpenAI

def encode_image(image: BytesIO) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image.read()).decode("utf-8")

def get_async_openai_client(model: str) -> AsyncOpenAI:
    """Initialize AsyncOpenAI client with model-specific base URL."""
    valid_models = ["gemma3", "gpt-oss"]
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Choose from: {', '.join(valid_models)}")
    
    model_ports = {
        "gemma3": "9000",
        "gpt-oss": "9500",
    }
    base_url = f"http://0.0.0.0:{model_ports[model]}/v1"
    ## TODO - Fix this hardcide 
    base_url = "https://<some-thing-here>.dwani.ai/v1"

    base_url = f"{os.getenv('DWANI_API_BASE_URL')}/v1"

    return AsyncOpenAI(api_key="http", base_url=base_url)


import argparse

if __name__ == "__main__":
    # Ensure EXTERNAL_API_BASE_URL is set
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)