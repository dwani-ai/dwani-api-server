
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

    base_url = f"{os.getenv('DWANI_API_BASE_URL')}"

    return AsyncOpenAI(api_key="http", base_url=base_url)





from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import uuid
import os
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import DATETIME
import enum

# -------------------------- Database Setup --------------------------

DATABASE_URL = "sqlite:///./files.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class FileStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileRecord(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, index=True)  # UUID as string
    filename = Column(String, index=True)
    content_type = Column(String)
    status = Column(SQLEnum(FileStatus), default=FileStatus.PENDING)
    extracted_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------- Pydantic Models --------------------------

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "pending"
    message: str = "File uploaded successfully. Extraction in progress."


class FileRetrieveResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    extracted_text: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# -------------------------- Background Extraction Task --------------------------

async def extract_and_store(file_id: str, pdf_bytes: bytes, filename: str, db: Session):
    db_file = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not db_file:
        return

    db_file.status = FileStatus.PROCESSING
    db.commit()

    try:
        # Reuse your existing extraction logic but with raw bytes
        images = await render_pdf_to_png_bytes(pdf_bytes)

        if not images:
            raise ValueError("No pages found in PDF")

        result = ""
        model = "gemma3"
        client = get_async_openai_client(model)

        for i, image in enumerate(images):
            image_bytes_io = BytesIO()
            image.save(image_bytes_io, format='JPEG', quality=85)
            image_bytes_io.seek(0)
            image_base64 = encode_image(image_bytes_io)

            single_message = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": f"Extract plain text from this single PDF page (page {i+1}). "
                                         "Preserve the original reading order, headings, lists, and paragraph structure. "
                                         "Output clean plain text only."}
            ]

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert OCR and document text extraction assistant. "
                                                  "Extract accurate, clean plain text from document images."},
                    {"role": "user", "content": single_message}
                ],
                temperature=0.2,
                max_tokens=2048
            )

            page_text = response.choices[0].message.content.strip()
            result += page_text + "\n\n"

        db_file.extracted_text = result.strip()
        db_file.status = FileStatus.COMPLETED

    except Exception as e:
        db_file.status = FileStatus.FAILED
        db_file.error_message = str(e)
        logger.error(f"Extraction failed for file {file_id}: {str(e)}")

    finally:
        db_file.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_file)


# Helper to convert bytes directly (avoid temp file)
from pdf2image import convert_from_bytes

async def render_pdf_to_png_bytes(pdf_bytes: bytes) -> List[Image.Image]:
    try:
        images = convert_from_bytes(pdf_bytes, fmt="png")
        return images
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise


# -------------------------- New Files Endpoints --------------------------

@app.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported currently.")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Read file content once
    content = await file.read()

    # Generate unique ID
    file_id = str(uuid.uuid4())

    # Save record in DB
    db_file = FileRecord(
        id=file_id,
        filename=file.filename,
        content_type=file.content_type,
        status=FileStatus.PENDING
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    # Schedule background extraction
    background_tasks.add_task(extract_and_store, file_id, content, file.filename, db)

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        message="File uploaded successfully. Extraction in progress."
    )


@app.get("/files/{file_id}", response_model=FileRetrieveResponse)
def get_file(file_id: str, db: Session = Depends(get_db)):
    db_file = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    return FileRetrieveResponse(
        file_id=db_file.id,
        filename=db_file.filename,
        status=db_file.status.value,
        extracted_text=db_file.extracted_text,
        error_message=db_file.error_message,
        created_at=db_file.created_at,
        updated_at=db_file.updated_at
    )


# Optional: list all files (for admin/debug)
@app.get("/files/")
def list_files(db: Session = Depends(get_db), limit: int = 20):
    files = db.query(FileRecord).order_by(FileRecord.created_at.desc()).limit(limit).all()
    return [
        {
            "file_id": f.id,
            "filename": f.filename,
            "status": f.status.value,
            "created_at": f.created_at
        } for f in files
    ]

import argparse

if __name__ == "__main__":
    # Ensure EXTERNAL_API_BASE_URL is set
    
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)