from fastapi import HTTPException
from ..models.requests import SUPPORTED_MODELS, SUPPORTED_LANGUAGES

def validate_model(model: str) -> str:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}. Must be one of {SUPPORTED_MODELS}")
    return model

def validate_language(lang: str, field_name: str) -> str:
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}: {lang}. Must be one of {SUPPORTED_LANGUAGES}")
    return lang