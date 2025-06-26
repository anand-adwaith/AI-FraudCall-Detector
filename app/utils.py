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
    
    # Add Gemma model variables
    _gemma_model = None
    _gemma_tokenizer = None
    _gemma_initialized = False
    
    # Add Llama model variables
    _llama_pipeline = None
    _llama_initialized = False

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
    
    @classmethod
    def initialize_gemma(cls, hf_token: str = "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"):
        """
        Initialize the Gemma model and tokenizer
        
        Args:
            hf_token: Hugging Face token for model access
        """
        if cls._gemma_initialized:
            logger.info("Gemma model already initialized")
            return
            
        try:
            logger.info("Initializing Gemma model and tokenizer...")
            
            # Import required libraries
            from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
            import torch
            import os
            
            # Set environment variables for Hugging Face
            os.environ["HF_READ_TOKEN"] = hf_token
            os.environ["HF_ENDPOINT"] = "https://huggingface.co"
            
            # Model configuration
            model_id = "google/gemma-3-1b-it"
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Initialize model
            cls._gemma_model = Gemma3ForCausalLM.from_pretrained(
                model_id, 
                quantization_config=quantization_config, 
                token=hf_token, 
                device_map="cuda"
            ).eval()
            
            # Initialize tokenizer
            cls._gemma_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token,
                device_map="cuda"
            )
            
            cls._gemma_initialized = True
            logger.info("Gemma model and tokenizer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemma model: {e}", exc_info=True)
            raise
    
    @classmethod
    def get_gemma_model_and_tokenizer(cls):
        """
        Get the Gemma model and tokenizer, initializing if necessary
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not cls._gemma_initialized:
            cls.initialize_gemma()
        
        return cls._gemma_model, cls._gemma_tokenizer
    
    @classmethod
    def initialize_llama(cls, hf_token: str = "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"):
        """
        Initialize the Llama 3.2 model pipeline
        
        Args:
            hf_token: Hugging Face token for model access
        """
        if cls._llama_initialized:
            logger.info("Llama model already initialized")
            return
            
        try:
            logger.info("Initializing Llama 3.2 model pipeline...")
            
            # Import required libraries
            from transformers import pipeline
            import torch
            
            # Model configuration
            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            
            # Initialize pipeline
            cls._llama_pipeline = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token
            )
            
            cls._llama_initialized = True
            logger.info("Llama 3.2 model pipeline initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {e}", exc_info=True)
            raise
    
    @classmethod
    def get_llama_pipeline(cls):
        """
        Get the Llama pipeline, initializing if necessary
        
        Returns:
            Llama pipeline
        """
        if not cls._llama_initialized:
            cls.initialize_llama()
        
        return cls._llama_pipeline