from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from ..logging_config import setup_logging

router = APIRouter(prefix="/v1", tags=["Utility"])
logger = setup_logging()

@router.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy", "model": "llm_model_name"}  # Placeholder model name

@router.get("/")
async def home():
    logger.debug("Redirecting to documentation")
    return RedirectResponse(url="/docs")