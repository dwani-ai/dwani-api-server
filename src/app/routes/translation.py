from fastapi import APIRouter, HTTPException
from ..config import settings
from ..logging_config import setup_logging
from ..services.external_api import call_external_api
from ..models.requests import TranslationRequest
from ..models.responses import TranslationResponse

router = APIRouter(prefix="/v1", tags=["Translation"])
logger = setup_logging()

@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    if not request.sentences:
        raise HTTPException(status_code=400, detail="Sentences cannot be empty")

    supported_languages = [
        "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
        "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
        "rus_Cyrl", "pol_Latn"
    ]
    if request.src_lang not in supported_languages or request.tgt_lang not in supported_languages:
        raise HTTPException(status_code=400, detail=f"Invalid language codes: src={request.src_lang}, tgt={request.tgt_lang}")

    logger.debug(f"Processing translation request: {len(request.sentences)} sentences", extra={
        "endpoint": "/v1/translate",
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang
    })

    external_url = f"{settings.DWANI_API_BASE_URL_TRANSLATE}/translate?src_lang={request.src_lang}&tgt_lang={request.tgt_lang}"
    payload = {"sentences": request.sentences}

    response = await call_external_api(external_url, payload=payload)
    translations = response.json().get("translations", [])
    if not translations or len(translations) != len(request.sentences):
        raise HTTPException(status_code=500, detail="Invalid response from translation service")
    logger.debug(f"Translation completed: {len(translations)} sentences")
    return TranslationResponse(translations=translations)