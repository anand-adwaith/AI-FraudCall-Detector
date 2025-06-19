import uvicorn
import json
import sys
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging

# find .env in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import routers
from app.routers.qa import router as qa_router
from app.utils import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scam-detector")

# Create FastAPI app
app = FastAPI(
    title="AI Fraud/Scam Detector API",
    description="API for detecting scams in call transcripts and text messages",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(qa_router)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up the server...")
    # Initialize models in background
    ModelManager.initialize()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

# Commented out the old main function
"""
def main():
    # Example query
    query = "Hello, Im bank manager of Canara bank, we noticed some unusual activity in your account, kindly share your otp to verify the transaction."
    top_k = 5

    # Test the RAG approach first
    print("\n=== Running Scam Classification with RAG CALL MODE===")
    print(f"Query (caller transcript):\n{query}\n")

    from app.services.rag_service import (
        initialize_rag_components, 
        run_rag_query,
        run_few_shot_query,
        get_llm_for_few_shot
    )

    retriever, llm = initialize_rag_components(top_k=top_k, mode="call")
    rag_response = run_rag_query(query=query, retriever=retriever, llm=llm, mode="call")

    print("\n--- Retrieved Context Chunks ---")
    for i, doc in enumerate(rag_response["results"], 1):
        print(f"\n[Doc {i}]")
        print(f"Score: {doc['score']}")
        print(f"Text: {doc['content']}")
        print(f"Metadata: {doc['metadata']}")
        
    print("\n--- LLM Classification Result (RAG) ---")
    print(json.dumps(rag_response["answer"], indent=2))
    
    # Now test the Few-Shot approach
    print("\n=== Running Scam Classification with Few-Shot Learning ===")
    print(f"Query (caller transcript):\n{query}\n")
    
    # Use the dedicated LLM for few-shot learning
    few_shot_llm = get_llm_for_few_shot(mode="call")
    few_shot_response = run_few_shot_query(query=query, llm=few_shot_llm, mode="call")
    
    print("\n--- LLM Classification Result (Few-Shot) ---")
    print(json.dumps(few_shot_response["answer"], indent=2))
    
    # Test with a text message example too
    text_query = "Warning: Your account has been locked due to suspicious activity. Click here to unlock: http://secure-account-verify.net"
    
    print("\n=== Running Text Scam Classification with Few-Shot Learning ===")
    print(f"Query (text message):\n{text_query}\n")
    
    text_few_shot_llm = get_llm_for_few_shot(mode="text")
    text_few_shot_response = run_few_shot_query(query=text_query, llm=text_few_shot_llm, mode="text")
    
    print("\n--- LLM Classification Result (Text Few-Shot) ---")
    print(json.dumps(text_few_shot_response["answer"], indent=2))
"""

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)