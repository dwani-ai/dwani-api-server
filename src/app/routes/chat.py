from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Form, Query
from time import time
from openai import AsyncOpenAI, OpenAIError
from ..config import settings
from ..logging_config import setup_logging
from ..services.external_api import call_external_api
from ..utils.validators import validate_model, validate_language
from ..models.requests import (
    ChatRequest, ChatDirectRequest, VisualQueryRequest, VisualQueryDirectRequest,
    ChatCompletionRequest
)
from ..models.responses import (
    ChatResponse, ChatDirectResponse, VisualQueryResponse, VisualQueryDirectResponse,
    ChatCompletionResponse, ChatCompletionChoice
)

router = APIRouter(prefix="/v1", tags=["Chat"])
logger = setup_logging()

openai_client = AsyncOpenAI(
    base_url=settings.DWANI_AI_LLM_URL,
    api_key=settings.DWANI_AI_LLM_API_KEY,
    timeout=30.0
)

@router.post("/indic_chat", response_model=ChatResponse)
async def indic_chat(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 1000 characters")

    validate_model(chat_request.model)
    validate_language(chat_request.src_lang, "source language")
    validate_language(chat_request.tgt_lang, "target language")

    logger.debug(f"Received indic chat request: prompt={chat_request.prompt[:50]}..., model={chat_request.model}", extra={
        "endpoint": "/v1/indic_chat",
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_LLM}/indic_chat"
    payload = {
        "prompt": chat_request.prompt,
        "src_lang": chat_request.src_lang,
        "tgt_lang": chat_request.tgt_lang,
        "model": chat_request.model
    }

    response = await call_external_api(external_url, payload=payload)
    response_text = response.json().get("response", "")
    logger.debug(f"Generated response: {response_text[:50]}...")
    return ChatResponse(response=response_text)

@router.post("/chat_direct", response_model=ChatDirectResponse)
async def chat_direct(request: Request, chat_request: ChatDirectRequest):
    if not chat_request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(chat_request.prompt) > 1000:
        raise HTTPException(status_code=400, detail="Prompt cannot exceed 1000 characters")

    validate_model(chat_request.model)

    logger.debug(f"Received direct chat request: prompt={chat_request.prompt[:50]}..., model={chat_request.model}", extra={
        "endpoint": "/v1/chat_direct",
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_LLM}/chat_direct"
    payload = {
        "prompt": chat_request.prompt,
        "model": chat_request.model,
        "system_prompt": chat_request.system_prompt
    }

    response = await call_external_api(external_url, payload=payload)
    response_text = response.json().get("response", "")
    logger.debug(f"Generated response: {response_text[:50]}...")
    return ChatDirectResponse(response=response_text)

@router.post("/indic_visual_query", response_model=VisualQueryResponse)
async def indic_visual_query(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    src_lang: str = Query(..., description="Source language code"),
    tgt_lang: str = Query(..., description="Target language code"),
    model: str = Query(default="gemma3", description="LLM model")
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 1000 characters")
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    validate_model(model)
    validate_language(src_lang, "source language")
    validate_language(tgt_lang, "target language")

    logger.debug("Processing indic visual query", extra={
        "endpoint": "/v1/indic_visual_query",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_VISION}/indic-visual-query/"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    data = {
        "prompt": query,
        "source_language": src_lang,
        "target_language": tgt_lang,
        "model": model
    }

    response = await call_external_api(external_url, files=files, data=data)
    answer = response.json().get("translated_response", "")
    if not answer:
        raise HTTPException(status_code=500, detail="No valid response provided by visual query service")
    logger.debug(f"Visual query response: {answer[:50]}...")
    return VisualQueryResponse(answer=answer)

@router.post("/visual_query_direct", response_model=VisualQueryDirectResponse)
async def visual_query_direct(
    request: Request,
    query: str = Form(..., description="Text query to describe or analyze the image"),
    file: UploadFile = File(..., description="Image file to analyze (PNG only)"),
    model: str = Query(default="gemma3", description="LLM model")
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(query) > 1000:
        raise HTTPException(status_code=400, detail="Query cannot exceed 1000 characters")
    if not file.content_type.startswith("image/png"):
        raise HTTPException(status_code=400, detail="Only PNG images supported")

    validate_model(model)

    logger.debug("Processing direct visual query", extra={
        "endpoint": "/v1/visual_query_direct",
        "query_length": len(query),
        "file_name": file.filename,
        "client_ip": request.client.host
    })

    external_url = f"{settings.DWANI_API_BASE_URL_VISION}/visual-query-direct/"
    file_content = await file.read()
    files = {"file": (file.filename, file_content, file.content_type)}
    data = {"prompt": query, "model": model}

    response = await call_external_api(external_url, files=files, data=data)
    answer = response.json().get("response", "")
    if not answer:
        raise HTTPException(status_code=500, detail="No valid response provided by visual query direct service")
    logger.debug(f"Direct visual query response: {answer[:50]}...")
    return VisualQueryDirectResponse(answer=answer)

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    logger.debug("Received chat completion request", extra={
        "endpoint": "/v1/chat/completions",
        "model": body.model,
        "client_ip": request.client.host
    })

    start_time = time()
    try:
        if body.stream:
            raise HTTPException(status_code=400, detail="Streaming not supported")

        response = await openai_client.chat.completions.create(
            model=body.model,
            messages=body.messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stream=body.stream
        )

        openai_response = ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[
                ChatCompletionChoice(
                    index=choice.index,
                    message={
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    finish_reason=choice.finish_reason
                ) for choice in response.choices
            ],
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            )
        )

        logger.debug(f"Chat completion completed in {time() - start_time:.2f} seconds")
        return openai_response

    except OpenAIError as e:
        status_code = 504 if "timeout" in str(e).lower() else 500
        logger.error(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=status_code, detail=f"LLM server error: {str(e)}")