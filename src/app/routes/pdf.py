from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
from time import time
from ..config import settings
from ..logging_config import setup_logging
from ..services.external_api import call_external_api
from ..utils.validators import validate_model, validate_language
from ..models.responses import (
    PDFTextExtractionResponse, DocumentProcessResponse, DocumentProcessPage,
    SummarizePDFResponse, IndicSummarizePDFResponse, CustomPromptPDFResponse,
    IndicCustomPromptPDFResponse
)

router = APIRouter(prefix="/v1", tags=["PDF"])
logger = setup_logging()

@router.post("/extract-text", response_model=PDFTextExtractionResponse)
async def extract_text(
    request: Request,
    file: UploadFile = File(..., description="PDF file to extract text from"),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)

    logger.debug("Processing PDF text extraction", extra={
        "endpoint": "/v1/extract-text",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/extract-text/"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    data = {"page_number": page_number, "model": model}

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    extracted_text = response.json().get("page_content", "") or ""
    logger.debug(f"Text extraction completed in {time() - start_time:.2f} seconds")
    return PDFTextExtractionResponse(page_content=extracted_text.strip())

@router.post("/indic-extract-text", response_model=DocumentProcessResponse)
async def extract_and_translate(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(1, description="Page number to extract text from (1-based indexing)", ge=1),
    src_lang: str = Form("eng_Latn", description="Source language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)
    validate_language(src_lang, "source language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic extract text", extra={
        "endpoint": "/v1/indic-extract-text",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/indic-extract-text/"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {
        "page_number": page_number,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model
    }

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    response_data = response.json()
    page = DocumentProcessPage(
        processed_page=response_data.get("processed_page", page_number),
        page_content=response_data.get("page_content", ""),
        translated_content=response_data.get("translated_content", "")
    )
    logger.debug(f"Indic extract text completed in {time() - start_time:.2f} seconds")
    return DocumentProcessResponse(pages=[page])

@router.post("/summarize-pdf", response_model=SummarizePDFResponse)
async def summarize_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(..., description="Page number to summarize (1-based indexing)", ge=1),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)

    logger.debug("Processing PDF summary", extra={
        "endpoint": "/v1/summarize-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/summarize-pdf"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {"page_number": page_number, "model": model}

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    response_data = response.json()
    result = SummarizePDFResponse(
        original_text=response_data.get("original_text", "No text extracted"),
        summary=response_data.get("summary", "No summary provided"),
        processed_page=response_data.get("processed_page", page_number)
    )
    logger.debug(f"PDF summary completed in {time() - start_time:.2f} seconds")
    return result

@router.post("/indic-summarize_pdf", response_model=IndicSummarizePDFResponse)
async def indic_summarize_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(..., description="Page number to summarize (1-based)", ge=1),
    src_lang: str = Form("eng_Latn", description="Source language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")

    validate_model(model)
    validate_language(src_lang, "source language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic summarize PDF", extra={
        "endpoint": "/v1/indic-summarize-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/indic-summarize-pdf"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {
        "page_number": page_number,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model
    }

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    response_data = response.json()
    result = IndicSummarizePDFResponse(
        original_text=response_data.get("original_text", "No text extracted"),
        summary=response_data.get("summary", "No summary provided"),
        translated_summary=response_data.get("translated_summary", "No translated summary provided"),
        processed_page=response_data.get("processed_page", page_number)
    )
    logger.debug(f"Indic PDF summary completed in {time() - start_time:.2f} seconds")
    return result

@router.post("/custom-prompt-pdf", response_model=CustomPromptPDFResponse)
async def custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt"),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    validate_model(model)

    logger.debug("Processing custom prompt PDF", extra={
        "endpoint": "/v1/custom-prompt-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client_ip
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/custom-prompt-pdf"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {"page_number": page_number, "prompt": prompt, "model": model}

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    response_data = response.json()
    result = CustomPromptPDFResponse(
        original_text=response_data.get("original_text", "No text extracted"),
        response=response_data.get("response", "No response provided"),
        processed_page=response_data.get("processed_page", page_number)
    )
    logger.debug(f"Custom prompt PDF completed in {time() - start_time:.2f} seconds")
    return result

@router.post("/indic-custom-prompt-pdf", response_model=IndicCustomPromptPDFResponse)
async def indic_custom_prompt_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(..., description="Page number to process (1-based indexing)", ge=1),
    prompt: str = Form(..., description="Custom prompt"),
    src_lang: str = Form("eng_Latn", description="Source language code"),
    tgt_lang: str = Form("kan_Knda", description="Target language code"),
    model: str = Form(default="gemma3", description="LLM model")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    validate_model(model)
    validate_language(src_lang, "source language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic custom prompt PDF", extra={
        "endpoint": "/v1/indic-custom-prompt-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/indic-custom-prompt-pdf"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {
        "page_number": str(page_number),
        "prompt": prompt,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "model": model
    }

    start_time = time()
    response = await call_external_api(external_url, files=files, data=data)
    response_data = response.json()
    result = IndicCustomPromptPDFResponse(
        original_text=response_data.get("original_text", "No text extracted"),
        response=response_data.get("response", "No response provided"),
        translated_response=response_data.get("translated_response", "No translated response provided"),
        processed_page=response_data.get("processed_page", page_number)
    )
    logger.debug(f"Indic custom prompt PDF completed in {time() - start_time:.2f} seconds")
    return result

@router.post("/indic-custom-prompt-kannada-pdf", responses={
    200: {"description": "Generated Kannada PDF file", "content": {"application/pdf": {}}},
    400: {"description": "Invalid PDF, page number, prompt, or language"},
    500: {"description": "External API error"},
    504: {"description": "External API timeout"}
})
async def indic_custom_prompt_kannada_pdf(
    request: Request,
    file: UploadFile = File(...),
    page_number: int = Form(..., description="Page number to process (1-based indexing)"),
    prompt: str = Form(..., description="Custom prompt"),
    src_lang: str = Form(..., description="Source language code"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be at least 1")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    supported_languages = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu"]
    if src_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {src_lang}")

    logger.debug("Processing Kannada PDF generation", extra={
        "endpoint": "/v1/indic-custom-prompt-kannada-pdf",
        "file_name": file.filename,
        "page_number": page_number,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_PDF}/indic-custom-prompt-kannada-pdf/"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, "application/pdf")}
    data = {
        "page_number": page_number,
        "prompt": prompt,
        "src_lang": src_lang
    }

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file_path = temp_file.name

    start_time = time()
    try:
        response = await call_external_api(external_url, files=files, data=data, stream=True)
        with open(temp_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        headers = {
            "Content-Disposition": "attachment; filename=\"generated_kannada.pdf\"",
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
        logger.debug(f"Kannada PDF generation completed in {time() - start_time:.2f} seconds")
        return FileResponse(
            path=temp_file_path,
            filename="generated_kannada.pdf",
            media_type="application/pdf",
            headers=headers
        )
    finally:
        temp_file.close()