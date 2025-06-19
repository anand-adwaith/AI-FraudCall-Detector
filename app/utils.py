import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scam-detector")

def load_fewshot_examples(csv_path, n=5):
    """Load examples from CSV file for few-shot learning."""
    df = pd.read_csv(csv_path)
    # Filter for strong scam/not-scam examples if desired
    # For now, just take the first n fraud and n normal
    fraud = df[df['Label'] == 'fraud'].head(n)
    normal = df[df['Label'] == 'normal'].head(n)
    intermediate = df[df['label'] == 'intermediate'].head(n) if 'intermediate' in df['Label'].unique() else pd.DataFrame()
    return pd.concat([fraud, normal, intermediate]).to_dict(orient='records')

def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent

def timer_decorator(func):
    """Decorator to time function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result
    return wrapper

class ModelManager:
    """Singleton class to manage model instances."""
    _instance = None
    _call_retriever = None
    _text_retriever = None
    _call_llm = None
    _text_llm = None
    _call_few_shot_llm = None 
    _text_few_shot_llm = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, top_k=5):
        """Initialize all models and retrievers if not already done."""
        if cls._initialized:
            logger.info("Models already initialized, skipping initialization")
            return
        
        from app.services.rag_service import (
            initialize_rag_components, 
            get_llm_for_few_shot
        )
        
        logger.info("Initializing models and retrievers...")
        
        # Initialize call components
        logger.info("Initializing call components...")
        cls._call_retriever, cls._call_llm = initialize_rag_components(top_k=top_k, mode="call")
        cls._call_few_shot_llm = get_llm_for_few_shot(mode="call")
        
        # Initialize text components
        logger.info("Initializing text components...")
        cls._text_retriever, cls._text_llm = initialize_rag_components(top_k=top_k, mode="text")
        cls._text_few_shot_llm = get_llm_for_few_shot(mode="text")
        
        cls._initialized = True
        logger.info("All models and retrievers initialized successfully!")
    
    @classmethod
    def get_components(cls, message_type="call", mode="rag"):
        """Get the appropriate components based on message type and mode."""
        if not cls._initialized:
            cls.initialize()
        
        if message_type == "call":
            if mode == "rag":
                return cls._call_retriever, cls._call_llm
            else:
                return None, cls._call_few_shot_llm
        else:  # text
            if mode == "rag":
                return cls._text_retriever, cls._text_llm
            else:
                return None, cls._text_few_shot_llm