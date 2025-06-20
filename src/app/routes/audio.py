from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, BackgroundTasks, Depends
from fastapi.responses import FileResponse, StreamingResponse
import tempfile
import os
from time import time
from ..config import settings
from ..logging_config import setup_logging
from ..services.tts_service import TTSService, get_tts_service
from ..services.external_api import call_external_api
from ..models.requests import SupportedLanguage
from ..models.responses import TranscriptionResponse

router = APIRouter(prefix="/v1", tags=["Audio"])
logger = setup_logging()

@router.post("/audio/speech", responses={
    200: {"description": "Audio file", "content": {"audio/mp3": {}}},
    400: {"description": "Invalid or empty input"},
    502: {"description": "External TTS service unavailable"},
    504: {"description": "TTS service timeout"}
})
async def generate_audio(
    request: Request,
    input: str = Query(..., description="Text to convert to speech (max 1000 characters)"),
    response_format: str = Query("mp3", description="Audio format (ignored, defaults to mp3)"),
    tts_service: TTSService = Depends(get_tts_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if len(input) > 1000:
        raise HTTPException(status_code=400, detail="Input cannot exceed 1000 characters")

    logger.debug("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(input),
        "client_ip": request.client.host
    })

    payload = {"text": input}
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name

    try:
        response = await tts_service.generate_speech(payload)
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
    finally:
        temp_file.close()

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil)")
):
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")

    start_time = time()
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    external_url = f"{settings.DWANI_API_BASE_URL_ASR}/transcribe/?language={language}"

    response = await call_external_api(external_url, files=files)
    transcription = response.json().get("text", "")
    logger.debug(f"Transcription completed in {time() - start_time:.2f} seconds")
    return TranscriptionResponse(text=transcription)

@router.post("/speech_to_speech", responses={
    200: {"description": "Audio stream", "content": {"audio/mp3": {}}},
    400: {"description": "Invalid input or language"},
    504: {"description": "External API timeout"},
    500: {"description": "External API error"}
})
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Query(..., description="Language of the audio (kannada, hindi, tamil)")
):
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")

    logger.debug("Processing speech-to-speech request", extra={
        "endpoint": "/v1/speech_to_speech",
        "audio_filename": file.filename,
        "language": language,
        "client_ip": request.client.host
    })

    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    external_url = f"{settings.DWANI_API_BASE_URL_S2S}/v1/speech_to_speech?language={language}"

    response = await call_external_api(external_url, files=files, stream=True)
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.mp3\"",
        "Cache-Control": "no-cache",
        "Content-Type": "audio/mp3"
    }
    return StreamingResponse(response.iter_content(chunk_size=8192), media_type="audio/mp3", headers=headers)