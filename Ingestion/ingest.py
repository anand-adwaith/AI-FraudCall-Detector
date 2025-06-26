# ingest.py

import os
import logging
from typing import List, Any
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from langchain.schema import Document
from langchain_qdrant import RetrievalMode,FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from preprocess import preprocess_dataframe
from app.config import (
    CSV_PATH,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    HF_MODEL_NAME,
    HF_MODEL_KWARGS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngestPipeline")


class ScamIngestPipeline:
    def __init__(self,
                 csv_path: str = "dataset/data.csv",
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "scam_db",
                 hf_model_name: str = "BAAI/bge-large-en-v1.5",
                 hf_model_kwargs: dict = None):
        """
        Initialize ingestion pipeline: loads config, sets up clients.
        """
        # Paths & names
        self.csv_path = csv_path
        self.collection_name = collection_name

        # HuggingFace embedding config
        self.hf_model_name = hf_model_name
        self.hf_model_kwargs = hf_model_kwargs or {"device": "cpu"}

        # Low-level Qdrant client still needed for collection management
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Placeholder for vector store
        self.vstore: QdrantVectorStore = None

        logger.info("ScamIngestPipeline initialized.")

    def preprocess(self) -> pd.DataFrame:
        """
        Load and clean the CSV, returning a DataFrame ready for ingestion.
        """
        logger.info(f"Loading data from {self.csv_path}...")
        df = preprocess_dataframe(self.csv_path, text_col="Text")
        logger.info(f"Preprocessing complete: {len(df)} valid rows.")
        return df

    def create_or_update_collection(self, dense_embedding_dim: int):
        """
        Create the Qdrant collection if not exists, else leave it.
        """
        existing = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name in existing:
            logger.info(f"Collection '{self.collection_name}' exists; skipping creation.")
        else:
            logger.info(f"Creating collection '{self.collection_name}' with dense_dim={dense_embedding_dim}...")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=dense_embedding_dim, distance=Distance.COSINE)
                }
            )
            logger.info("Collection created.")

    def ingest(self, batch_size: int = 64):
        """
        Full ingestion: preprocess, load documents, embed, and upsert into Qdrant.
        """
        df = self.preprocess()

        logger.info("Constructing Document objects with metadata...")
        documents = []
        documents: List[Document] = [
            Document(page_content=row["Text"],
                     metadata={"label": row["Label"], "scam_category": row["Type"]})
            for _, row in df.iterrows()
        ]

        logger.info(f"{len(documents)} documents constructed.")

        logger.info("Initializing HuggingFace embeddings...")
        dense_embeddings = HuggingFaceEmbeddings(
            model_name=self.hf_model_name,
            model_kwargs=self.hf_model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        
        dense_embedding_dimension = len(dense_embeddings.embed_query("I just want to know the embedding space dimension"))
        # Create or ensure collection
        self.create_or_update_collection(dense_embedding_dim=dense_embedding_dimension)

        logger.info("Connecting LangChain Qdrant vector store...")
        logger.info("Upserting documents into Qdrant...")
        self.vstore = QdrantVectorStore(
            embedding=dense_embeddings,
            client = self.qdrant_client,
            collection_name=self.collection_name,
            vector_name="dense",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
        )
        self.vstore.add_documents(documents, batch_size=batch_size)
        logger.info("Ingestion complete.")

    def retrieve_top_k(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Connect to Qdrant DB and collection separately, then retrieve top_k.
        """
        # ensure a fresh connection to QdrantVectorStore
        embeddings = HuggingFaceEmbeddings(
            model_name=self.hf_model_name,
            model_kwargs=self.hf_model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        # connect to vector collection
        store = QdrantVectorStore(
            embedding=embeddings,
            client=self.qdrant_client,
            collection_name=self.collection_name,
            vector_name="dense",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
        )
        # perform dense retrieval
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        results = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(results)} documents for query '{query}'.")

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, "score", None),
            }
            for doc in results
        ]


if __name__ == "__main__":
    pipeline = ScamIngestPipeline(
        csv_path=CSV_PATH,
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
        hf_model_name=HF_MODEL_NAME,
        hf_model_kwargs=HF_MODEL_KWARGS,
    )
    pipeline.ingest()
    sample = pipeline.retrieve_top_k("share the OTP received", top_k=3)
    for idx, r in enumerate(sample, start=1):
        print(f"\nResult #{idx}")
        print("Score:", r["score"])
        print("Content:", r["content"])
        print("Metadata:", r["metadata"])
