from fastapi import APIRouter, HTTPException
import asyncio
import logging
import os
from typing import Dict, Any
from app.models.speech import (
    TranscribeRequest,
    TranscribeResponse,
    TranslateRequest,
    TranslateResponse,
    TranscribeAndTranslateRequest,
    TranscribeAndTranslateResponse
)
from app.services.asr import initialize_asr_model, asr_transcribe
from app.services.translate import (
    initialize_translate_tokenizer_and_model,
    translate_indic_to_english,
    transcribe_and_translate
)

# Configure logging
logger = logging.getLogger("speech-router")

# Global model variables
ASR_MODEL = None
TRANSLATE_MODEL = None
TRANSLATE_TOKENIZER = None

# Create router
router = APIRouter(prefix="/api", tags=["Speech Processing"])

def initialize_models():
    """Initialize all models needed for speech processing"""
    global ASR_MODEL, TRANSLATE_MODEL, TRANSLATE_TOKENIZER
    
    try:
        logger.info("Initializing ASR model for speech router...")
        ASR_MODEL = initialize_asr_model()
        logger.info("ASR model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ASR model: {str(e)}")
        raise
    
    try:
        logger.info("Initializing translation model and tokenizer for speech router...")
        TRANSLATE_TOKENIZER, TRANSLATE_MODEL = initialize_translate_tokenizer_and_model()
        logger.info("Translation model and tokenizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize translation model: {str(e)}")
        raise
    
    return {"asr_model": ASR_MODEL, "translate_tokenizer": TRANSLATE_TOKENIZER, "translate_model": TRANSLATE_MODEL}

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: TranscribeRequest):
    """
    Transcribe an audio file using ASR.
    
    Args:
        request: TranscribeRequest with file_path and language_id
        
    Returns:
        TranscribeResponse with transcription text or error
    """
    global ASR_MODEL
    
    try:
        # Check if model is initialized
        if ASR_MODEL is None:
            logger.warning("ASR model not initialized, initializing now")
            ASR_MODEL = initialize_asr_model()
            
        if ASR_MODEL is None:
            raise ValueError("Failed to initialize ASR model")
        
        # Set timeout for operation
        transcription = await asyncio.wait_for(
            asr_transcribe(ASR_MODEL, request.file_path, request.language_id),
            timeout=100  # 100 seconds timeout
        )
        
        return TranscribeResponse(text=transcription)
    
    except asyncio.TimeoutError:
        logger.error(f"Transcription timed out after 100 seconds for {request.file_path}")
        return TranscribeResponse(error="Transcription timed out after 100 seconds")
    
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {str(e)}")
        return TranscribeResponse(error=f"Audio file not found: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return TranscribeResponse(error=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return TranscribeResponse(error=f"Transcription failed: {str(e)}")

@router.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate text from an Indic language to English.
    
    Args:
        request: TranslateRequest with text and language_id
        
    Returns:
        TranslateResponse with translation text or error
    """
    global TRANSLATE_TOKENIZER, TRANSLATE_MODEL
    
    try:
        # Check if models are initialized
        if TRANSLATE_TOKENIZER is None or TRANSLATE_MODEL is None:
            logger.warning("Translation models not initialized, initializing now")
            TRANSLATE_TOKENIZER, TRANSLATE_MODEL = initialize_translate_tokenizer_and_model()
            
        if TRANSLATE_TOKENIZER is None or TRANSLATE_MODEL is None:
            raise ValueError("Failed to initialize translation models")
        
        # Validate input
        if not request.text or not request.text.strip():
            raise ValueError("Empty or invalid text input")
            
        if not request.language_id:
            raise ValueError("Missing language ID")
        
        # Set timeout for operation
        translation = await asyncio.wait_for(
            translate_indic_to_english(request.text, request.language_id, TRANSLATE_TOKENIZER, TRANSLATE_MODEL),
            timeout=100  # 100 seconds timeout
        )
        
        return TranslateResponse(text=translation)
    
    except asyncio.TimeoutError:
        logger.error(f"Translation timed out after 100 seconds")
        return TranslateResponse(error="Translation timed out after 100 seconds")
    
    except ValueError as e:
        logger.error(f"Invalid input for translation: {str(e)}")
        return TranslateResponse(error=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        return TranslateResponse(error=f"Translation failed: {str(e)}")

@router.post("/transcribe_and_translate", response_model=TranscribeAndTranslateResponse)
async def process_audio_to_english(request: TranscribeAndTranslateRequest):
    """
    Transcribe an audio file and translate the transcription to English.
    
    Args:
        request: TranscribeAndTranslateRequest with file_path and language_id
        
    Returns:
        TranscribeAndTranslateResponse with transcription text, translation text, or error
    """
    global ASR_MODEL, TRANSLATE_TOKENIZER, TRANSLATE_MODEL
    
    try:
        # Check if models are initialized
        if ASR_MODEL is None:
            logger.warning("ASR model not initialized, initializing now")
            ASR_MODEL = initialize_asr_model()
            
        if TRANSLATE_TOKENIZER is None or TRANSLATE_MODEL is None:
            logger.warning("Translation models not initialized, initializing now")
            TRANSLATE_TOKENIZER, TRANSLATE_MODEL = initialize_translate_tokenizer_and_model()
            
        if ASR_MODEL is None or TRANSLATE_TOKENIZER is None or TRANSLATE_MODEL is None:
            raise ValueError("Failed to initialize required models")
        
        # Validate input
        if not request.file_path or not os.path.exists(request.file_path):
            raise FileNotFoundError(f"Audio file not found: {request.file_path}")
            
        if not request.language_id:
            raise ValueError("Missing language ID")
        
        # Implement our own transcribe_and_translate to use the global models
        # First, transcribe
        transcription = await asyncio.wait_for(
            asr_transcribe(ASR_MODEL, request.file_path, request.language_id),
            timeout=50  # 50 seconds timeout (half of the total to allow for translation)
        )
        
        # Then, translate
        translation = await asyncio.wait_for(
            translate_indic_to_english(transcription, request.language_id, TRANSLATE_TOKENIZER, TRANSLATE_MODEL),
            timeout=50  # 50 seconds timeout (half of the total)
        )
        return TranscribeAndTranslateResponse(
            transcription=transcription,
            translation=translation
            )
    except asyncio.TimeoutError:
        logger.error(f"Process timed out after 100 seconds for {request.file_path}")
        return TranscribeAndTranslateResponse(error="Process timed out after 100 seconds")
        
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {str(e)}")
        return TranscribeAndTranslateResponse(error=f"Audio file not found: {str(e)}")
        
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        return TranscribeAndTranslateResponse(error=f"Invalid input: {str(e)}")
    
    except Exception as e:
        logger.error(f"Process failed: {str(e)}", exc_info=True)
        return TranscribeAndTranslateResponse(error=f"Process failed: {str(e)}")
    
    except asyncio.TimeoutError:
        logger.error(f"Process timed out after 100 seconds for {request.file_path}")
        return TranscribeAndTranslateResponse(
            error="Process timed out after 100 seconds"
        )
    
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        return TranscribeAndTranslateResponse(
            error=f"Process failed: {str(e)}"
        )
