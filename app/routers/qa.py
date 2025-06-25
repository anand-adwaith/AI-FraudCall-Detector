from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, File, UploadFile, Form
from typing import Dict, Any, Optional
import asyncio
import logging
import os
from app.models.schema import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse,
    ClassificationResponse,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioClassificationRequest,
    AudioClassificationResponse
)
from app.utils import ModelManager, timer_decorator

logger = logging.getLogger("scam-detector")

router = APIRouter(prefix="/api", tags=["Scam Detection"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "message": "API is running"}

@router.post("/initialize")
async def initialize_models(background_tasks: BackgroundTasks):
    """Initialize models and retrievers in background."""
    
    # Perform initialization in the background to not block the response
    background_tasks.add_task(ModelManager.initialize)
    
    return {"message": "Model initialization started in background"}

@timer_decorator
@router.post("/analyze", response_model=QueryResponse)
async def analyze_query(request: QueryRequest):
    """
    Analyze a query for scam detection.
    
    This endpoint supports both call transcripts and text messages,
    and can use either RAG or Few-Shot analysis modes.
    """
    try:
        # Get appropriate components based on request
        retriever, llm = ModelManager.get_components(
            message_type=request.message_type.value,
            mode=request.mode.value
        )
          # Choose the function to use based on mode
        from app.services.rag_service import run_rag_query, run_few_shot_query
          # Check if using Gemma model
        if request.model_type.value.lower() == "gemma":
            # No need to initialize Gemma here, just verify it was initialized at startup
            if not ModelManager._gemma_initialized:
                logger.error("Requested Gemma model but it was not initialized during startup")
                raise HTTPException(status_code=500, detail="Gemma model is not available. Please check server logs.")
            logger.info("Using pre-initialized Gemma model for inference")
        
        # Check if using Llama model
        elif request.model_type.value.lower() == "llama":
            # No need to initialize Llama here, just verify it was initialized at startup
            if not ModelManager._llama_initialized:
                logger.error("Requested Llama model but it was not initialized during startup")
                raise HTTPException(status_code=500, detail="Llama model is not available. Please check server logs.")
            logger.info("Using pre-initialized Llama model for inference")
        
        # Set timeout for the operation (60 seconds)
        if request.mode.value == "rag":
            # Use RAG mode
            logger.info(f"Running {request.message_type.value} analysis in RAG mode with {request.model_type.value} model")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_rag_query,
                        query=request.query,
                        retriever=retriever,
                        llm=llm,
                        mode=request.message_type.value,
                        model_type=request.model_type.value
                    ),
                    timeout=200
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Analysis timed out after 60 seconds")
                
        else:  # Few-shot mode
            logger.info(f"Running {request.message_type.value} analysis in Few-Shot mode with {request.model_type.value} model")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_few_shot_query,
                        query=request.query,
                        llm=llm,
                        mode=request.message_type.value,
                        model_type=request.model_type.value
                    ),
                    timeout=200
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Analysis timed out after 60 seconds")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@timer_decorator
@router.post("/audio-classify", response_model=AudioClassificationResponse)
async def classify_audio(request: AudioClassificationRequest):
    """
    Process an audio file through transcription, translation, and then classification.
    
    This endpoint:
    1. Transcribes the audio file in the specified language
    2. Translates the transcription to English
    3. Runs scam/fraud detection using either RAG or Few-Shot technique
    4. Returns the transcription, translation, and classification results
    """
    try:
        # Access models from the speech router
        from app.routers.speech import ASR_MODEL, TRANSLATE_MODEL, TRANSLATE_TOKENIZER
        
        if not ASR_MODEL:
            from app.services.asr import initialize_asr_model
            ASR_MODEL = initialize_asr_model()
            logger.info("ASR model initialized on-demand")
            
        if not TRANSLATE_MODEL or not TRANSLATE_TOKENIZER:
            from app.services.translate import initialize_translate_tokenizer_and_model
            TRANSLATE_TOKENIZER, TRANSLATE_MODEL = initialize_translate_tokenizer_and_model()
            logger.info("Translation models initialized on-demand")
          # Step 1: Transcribe the audio
        logger.info(f"Transcribing audio file: {request.file_path} (language: {request.language_id})")
        
        # Validate file path
        if not os.path.exists(request.file_path):
            return AudioClassificationResponse(error=f"Audio file not found: {request.file_path}")
            
        from app.services.asr import asr_transcribe
        
        try:
            # Set timeout for transcription (40 seconds)
            transcription = await asyncio.wait_for(
                asr_transcribe(ASR_MODEL, request.file_path, request.language_id),
                timeout=40
            )
            logger.info(f"Transcription successful: {transcription[:100]}...")
        except asyncio.TimeoutError:
            return AudioClassificationResponse(error="Transcription timed out after 40 seconds")
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return AudioClassificationResponse(error=f"Transcription failed: {str(e)}")
        
        # Step 2: Translate to English
        logger.info(f"Translating transcription to English")
        from app.services.translate import translate_indic_to_english
        
        try:
            # Set timeout for translation (30 seconds)
            translation = await asyncio.wait_for(
                translate_indic_to_english(
                    transcription, 
                    request.language_id, 
                    TRANSLATE_TOKENIZER, 
                    TRANSLATE_MODEL
                ),
                timeout=30
            )
            logger.info(f"Translation successful: {translation[:100]}...")
        except asyncio.TimeoutError:
            return AudioClassificationResponse(
                transcription=transcription,
                error="Translation timed out after 30 seconds"
            )
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return AudioClassificationResponse(
                transcription=transcription,
                error=f"Translation failed: {str(e)}"
            )
        
        # Step 3: Get appropriate components based on request
        from app.utils import ModelManager
        retriever, llm = ModelManager.get_components(
            message_type=request.model_type.value,
            mode=request.analysis_type.value
        )
          # Step 4: Choose the function to use based on mode and run classification
        from app.services.rag_service import run_rag_query, run_few_shot_query
        
        try:            # Initialize Gemma model if requested
            if request.llm_type.value.lower() == "gemma":
                try:
                    # Initialize Gemma model if not already done
                    from app.utils import ModelManager
                    ModelManager.initialize_gemma()
                    logger.info("Using Gemma model for inference")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemma model: {str(e)}")
                    return AudioClassificationResponse(
                        transcription=transcription,
                        translation=translation,
                        error=f"Failed to initialize Gemma model: {str(e)}"
                    )
            
            # Initialize Llama model if requested
            elif request.llm_type.value.lower() == "llama":
                try:
                    # Initialize Llama model if not already done
                    from app.utils import ModelManager
                    ModelManager.initialize_llama()
                    logger.info("Using Llama model for inference")
                except Exception as e:
                    logger.error(f"Failed to initialize Llama model: {str(e)}")
                    return AudioClassificationResponse(
                        transcription=transcription,
                        translation=translation,
                        error=f"Failed to initialize Llama model: {str(e)}"
                    )
            
            # Set timeout for classification (30 seconds)
            if request.analysis_type.value == "rag":
                # Use RAG mode
                logger.info(f"Running {request.model_type.value} analysis in RAG mode with {request.llm_type.value} model")
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_rag_query,
                        query=translation,
                        retriever=retriever,
                        llm=llm,
                        mode=request.model_type.value,
                        model_type=request.llm_type.value
                    ),
                    timeout=30
                )
            else:
                # Use Few-shot mode
                logger.info(f"Running {request.model_type.value} analysis in Few-Shot mode with {request.llm_type.value} model")
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_few_shot_query,
                        query=translation,
                        llm=llm,
                        mode=request.model_type.value,
                        model_type=request.llm_type.value
                    ),
                    timeout=30
                )
                
            # Return the complete result
            return AudioClassificationResponse(
                transcription=transcription,
                translation=translation,
                classification=response["answer"]
            )
            
        except asyncio.TimeoutError:
            return AudioClassificationResponse(
                transcription=transcription,
                translation=translation,
                error="Classification timed out after 30 seconds"
            )
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return AudioClassificationResponse(
                transcription=transcription,
                translation=translation,
                error=f"Classification failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Audio classification failed: {str(e)}", exc_info=True)
        return AudioClassificationResponse(error=f"Audio classification failed: {str(e)}")

