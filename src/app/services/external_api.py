import requests
from fastapi import HTTPException
from ..config import settings
from ..logging_config import setup_logging

logger = setup_logging()

async def call_external_api(endpoint: str, payload: dict = None, files: dict = None, data: dict = None, stream: bool = False, timeout: int = 60):
    try:
        headers = {"accept": "application/json", "Content-Type": "application/json"} if not files else {"accept": "application/json"}
        response = requests.post(
            endpoint,
            json=payload,
            files=files,
            data=data,
            headers=headers,
            stream=stream,
            timeout=timeout
        )
        response.raise_for_status()
        return response
    except requests.Timeout:
        logger.error(f"External API request timed out: {endpoint}")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from external API")