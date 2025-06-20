from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

class ChatDirectResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

class VisualQueryResponse(BaseModel):
    answer: str = Field(..., description="Visual query response")

class VisualQueryDirectResponse(BaseModel):
    answer: str = Field(..., description="Direct visual query response")

class PDFTextExtractionResponse(BaseModel):
    page_content: str = Field(..., description="Extracted text from the specified PDF page")

class DocumentProcessPage(BaseModel):
    processed_page: int = Field(..., description="Page number of the extracted text")
    page_content: str = Field(..., description="Extracted text from the page")
    translated_content: Optional[str] = Field(None, description="Translated text of the page, if applicable")

class DocumentProcessResponse(BaseModel):
    pages: List[DocumentProcessPage] = Field(..., description="List of pages with extracted and translated text")

class SummarizePDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    summary: str = Field(..., description="Summary of the specified page")
    processed_page: int = Field(..., description="Page number processed")

class IndicSummarizePDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    summary: str = Field(..., description="Summary of the specified page in the source language")
    translated_summary: str = Field(..., description="Summary translated into the target language")
    processed_page: int = Field(..., description="Page number processed")

class CustomPromptPDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    response: str = Field(..., description="Response based on the custom prompt")
    processed_page: int = Field(..., description="Page number processed")

class IndicCustomPromptPDFResponse(BaseModel):
    original_text: str = Field(..., description="Extracted text from the specified page")
    response: str = Field(..., description="Response based on the custom prompt")
    translated_response: str = Field(..., description="Translated response in the target language")
    processed_page: int = Field(..., description="Page number processed")

class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None