from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

SUPPORTED_MODELS = ["gemma3", "moondream", "qwen2.5vl", "qwen3", "sarvam-m", "deepseek-r1"]
SUPPORTED_LANGUAGES = ["kan_Knda", "hin_Deva", "tam_Taml", "tel_Telu", "eng_Latn", "deu_Latn"]

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 1000 characters)", max_length=1000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class ChatDirectRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for chat (max 1000 characters)", max_length=1000)
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)
    system_prompt: str = Field(default="", description="System prompt")

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class VisualQueryDirectRequest(BaseModel):
    query: str = Field(..., description="Text query", max_length=1000)
    model: str = Field(default="gemma3", description="LLM model", enum=SUPPORTED_MODELS)

class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemma-3-12b-it", description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")