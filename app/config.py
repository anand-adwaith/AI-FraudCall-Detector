import os

# Dataset path
CSV_PATH = os.getenv("CSV_PATH", "dataset/merged_call_text.csv")

# Qdrant vector DB
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scam_db")
# CALL_QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scam_db_call")
# TEXT_QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "scam_db_text")

# HuggingFace embedding model
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "BAAI/bge-large-en-v1.5")
HF_MODEL_KWARGS = {
    "device": os.getenv("HF_DEVICE", "cpu")  # or "cuda" if using GPU
}

# Embedding vector configuration (if you want to keep fixed sizes)
DENSE_VECTOR_NAME = "dense"
DENSE_DISTANCE_METRIC = "COSINE"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")  # or the version you use
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")  # e.g., "gpt-35-turbo" or "gpt-4"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBwensyPgaAPJffJ57a-b5NVmwsDmfbCjs")
