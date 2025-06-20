from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class TranscribeRequest(BaseModel):
    file_path: str = Field(..., description="Path to the audio file")
    language_id: str = Field(..., description="Language ID for transcription (e.g., 'hi' for Hindi)")

class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    language_id: str = Field(..., description="Source language ID (e.g., 'hi' for Hindi)")

class TranscribeResponse(BaseModel):
    text: str = Field(None, description="The transcribed text")
    error: Optional[str] = None

class TranslateResponse(BaseModel):
    text: str = Field(None, description="The translated text")
    error: Optional[str] = None

class TranscribeAndTranslateRequest(BaseModel):
    file_path: str = Field(..., description="Path to the audio file")
    language_id: str = Field(..., description="Language ID for transcription (e.g., 'hi' for Hindi)")

class TranscribeAndTranslateResponse(BaseModel):
    transcription: str = Field(None, description="The transcribed text")
    translation: str = Field(None, description="The translated text")
    error: Optional[str] = None
