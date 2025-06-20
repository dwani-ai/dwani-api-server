from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DWANI_API_BASE_URL_PDF: str
    DWANI_API_BASE_URL_VISION: str
    DWANI_API_BASE_URL_LLM: str
    DWANI_API_BASE_URL_LLM_QWEN: str
    DWANI_API_BASE_URL_TTS: str
    DWANI_API_BASE_URL_ASR: str
    DWANI_API_BASE_URL_TRANSLATE: str
    DWANI_API_BASE_URL_S2S: str
    DWANI_AI_LLM_URL: str
    DWANI_AI_LLM_API_KEY: str = ""
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()