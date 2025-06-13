from abc import ABC, abstractmethod
from fastapi import HTTPException
import requests
from ..config import settings
from ..logging_config import setup_logging

logger = setup_logging()

class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> requests.Response:
        pass

class ExternalTTSService(TTSService):
    async def generate_speech(self, payload: dict) -> requests.Response:
        try:
            base_url = f"{settings.DWANI_API_BASE_URL_TTS}/v1/audio/speech"
            return requests.post(
                base_url,
                json=payload,
                headers={"accept": "*/*", "Content-Type": "application/json"},
                stream=True,
                timeout=60
            )
        except requests.Timeout:
            logger.error("External TTS API timeout")
            raise HTTPException(status_code=504, detail="External TTS API timeout")
        except requests.RequestException as e:
            logger.error(f"External TTS API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"External TTS service error: {str(e)}")

def get_tts_service() -> TTSService:
    return ExternalTTSService()