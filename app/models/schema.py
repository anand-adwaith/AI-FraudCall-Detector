from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class AnalysisMode(str, Enum):
    RAG = "rag"
    FEW_SHOT = "few_shot"

class MessageType(str, Enum):
    CALL = "call"
    TEXT = "text"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The text message or call transcript to analyze")
    message_type: MessageType = Field(..., description="Whether the query is a call transcript or text message")
    mode: AnalysisMode = Field(default=AnalysisMode.RAG, description="Analysis mode: RAG or few_shot")
    top_k: Optional[int] = Field(default=5, description="Number of similar documents to retrieve in RAG mode")

class DocumentResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class ClassificationResponse(BaseModel):
    classification: str = Field(..., description="Scam / Not Scam / Suspicious")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(..., description="Detailed reasoning for the classification")
    follow_up_questions: Optional[List[str]] = Field(default=[], description="Follow-up questions for suspicious cases")

class QueryResponse(BaseModel):
    results: Optional[List[DocumentResult]] = Field(default=[], description="Retrieved documents (for RAG mode only)")
    answer: ClassificationResponse = Field(..., description="Classification results")
    
class HealthResponse(BaseModel):
    status: str
    message: str
