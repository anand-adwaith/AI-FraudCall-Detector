from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional
import asyncio
import logging
from app.models.schema import (
    QueryRequest, 
    QueryResponse, 
    HealthResponse,
    ClassificationResponse
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
        
        # Set timeout for the operation (60 seconds)
        if request.mode.value == "rag":
            # Use RAG mode
            logger.info(f"Running {request.message_type.value} analysis in RAG mode")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_rag_query,
                        query=request.query,
                        retriever=retriever,
                        llm=llm,
                        mode=request.message_type.value
                    ),
                    timeout=60
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Analysis timed out after 60 seconds")
                
        else:  # Few-shot mode
            logger.info(f"Running {request.message_type.value} analysis in Few-Shot mode")
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_few_shot_query,
                        query=request.query,
                        llm=llm,
                        mode=request.message_type.value
                    ),
                    timeout=60
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Analysis timed out after 60 seconds")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
