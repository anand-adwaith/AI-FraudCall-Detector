import os

# Dataset path
CSV_PATH = os.getenv("CSV_PATH", "dataset/merged_text_data.csv")

# Qdrant vector DB
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CALL_QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scam_db_call")
TEXT_QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scam_db_text")

# HuggingFace embedding model
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "BAAI/bge-large-en-v1.5")
HF_MODEL_KWARGS = {
    "device": os.getenv("HF_DEVICE", "cpu")  # or "cuda" if using GPU
}

# Embedding vector configuration (if you want to keep fixed sizes)
DENSE_VECTOR_NAME = "dense"
DENSE_DISTANCE_METRIC = "COSINE"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "dLbxCKdmrveV42xGDQQKOZ7PEnoXtjTASJET6JK1cdovLAk7WCvtJQQJ99BEACHYHv6XJ3w3AAAAACOG8UMJ")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://mha2c-mabd2a4o-eastus2.cognitiveservices.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")  # or the version you use
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")  # e.g., "gpt-35-turbo" or "gpt-4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBwensyPgaAPJffJ57a-b5NVmwsDmfbCjs")
