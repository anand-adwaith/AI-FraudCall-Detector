from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools.convert_to_openai import format_tool_to_openai_function
from langchain.schema import Document
from dotenv import load_dotenv
import os
from pathlib import Path
import json
# find .env in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.tools.convert_to_openai import format_tool_to_openai_function

from app.config import (
    QDRANT_URL,
    CALL_QDRANT_COLLECTION_NAME,
    TEXT_QDRANT_COLLECTION_NAME,
    HF_MODEL_NAME,
    HF_MODEL_KWARGS,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)

# --------------------------
# Function (tool) definition
# --------------------------
# @tool
# def classify_scam_call(
#     classification: str,
#     confidence: float,
#     reasoning: str,
#     follow_up_questions: list = []
# ) -> dict:
#     """Used by AI to classify if a given user query is a Scam / Not Scam / Suspicious
#     based on surrounding context, and provide a confidence score and rationale."""
#     return {
#         "classification": classification,
#         "confidence": confidence,
#         "reasoning": reasoning,
#         "follow_up_questions": follow_up_questions,
#     }

classify_scam_call_schema = {
    "name": "classify_scam_call",
    "description": "Used by AI to classify if a given user query is a Scam / Not Scam / Suspicious based on surrounding context, and provide a confidence score and rationale.",
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "description": "Scam/Not Scam/Suspicious"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1"
            },
            "reasoning": {
                "type": "string",
                "description": "Reason for classification"
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Follow-up questions if Suspicious"
            }
        },
        "required": ["classification", "confidence", "reasoning"]
    }
}

classify_scam_text_schema = {
    "name": "classify_scam_text",
    "description": "Used by AI to classify if a given user query is a Scam / Not Scam / Suspicious based on surrounding context, and provide a confidence score and rationale.",
    "parameters": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "description": "Scam/Not Scam/Suspicious"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1"
            },
            "reasoning": {
                "type": "string",
                "description": "Reason for classification"
            },
        },
        "required": ["classification", "confidence", "reasoning"]
    }
}

# --------------------------
# Prompt Setup
# --------------------------
SYSTEM_PROMPT_CALL_RAG = """
You are an AI Fraud/Scam Call Detector.

You will be given a transcript of what a caller (potential scammer) said during a phone call. 
Your task is to analyze this caller transcript using context documents retrieved from a database. 
These retrieved messages include previous scam and non-scam examples, with labels and scam categories.

Use this retrieved information to:

1. **Classify** the caller's behavior as one of:
   - "Scam"
   - "Not Scam"
   - "Suspicious"

2. **Provide a confidence score** between 0 and 1 for your classification.

3. **Give clear reasoning** for your decision based on the caller's transcript and context.

4. **If classified as 'Suspicious'**, provide 2–3 follow-up questions that the user can ask the caller 
   to help determine if it's a scam (e.g., asking for identity proof, verification steps, or refusal to share sensitive info).

You MUST respond using the function `classify_scam_call_schema` with the correct JSON structure.
Only use the transcript and context; do not make assumptions beyond the data.
Be precise and helpful.
"""

SYSTEM_PROMPT_TEXT_RAG = """
You are an AI Fraud/Scam Text Detector.

You will be given a transcript of what a (potential scammer) sent as a phone text message. 
Your task is to analyze this text transcript using context documents retrieved from a database. 
These retrieved messages include previous scam and non-scam examples, with labels and scam categories.

Use this retrieved information to:

1. **Classify** the text message as one of:
   - "Scam"
   - "Not Scam"
   - "Suspicious"

2. **Provide a confidence score** between 0 and 1 for your classification.

3. **Give clear reasoning** for your decision based on the text message and context.

You MUST respond using the function `classify_scam_text_schema` with the correct JSON structure.
Only use the transcript and context; do not make assumptions beyond the data.
Be precise and helpful.
"""

# --------------------------
# Few-Shot Prompt Setup
# --------------------------
SYSTEM_PROMPT_CALL_FEW_SHOT = """
You are an AI Fraud/Scam Call Detector.

You will be given a transcript of what a caller (potential scammer) said during a phone call.
Your task is to analyze this transcript and determine if it represents a scam call.

I'll provide you with several examples to help you understand different types of scam patterns. For each example, I'll show the thought process for classification.

## EXAMPLES OF SCAMS AND THOUGHT PROCESSES:

### Example 1: Phishing
Transcript: "Dear customer, your Paytm wallet has been blocked for security reasons. Please confirm your PAN details to reactivate your account."
Thought Process: This transcript has multiple red flags:
1. Creates urgency with account blocking
2. Asks for sensitive PAN details
3. Doesn't specify how the customer will be identified
4. Uses vague "Dear customer" greeting rather than person's name
5. No official verification channels mentioned
Classification: Scam
Confidence: 0.95
Reasoning: This is a clear phishing attempt targeting financial information. Legitimate companies don't ask for complete PAN details over calls. The urgency tactic is designed to prevent critical thinking.
Follow-up questions:
- "Can I call the official Paytm customer service to verify this issue?"
- "What's the official Paytm helpline number I should be contacting?"
- "How did you identify me as a customer without verifying my identity first?"

### Example 2: Identity Theft
Transcript: "Hello, this is Anjali from the Income Tax Department. There is an issue with your recent filing. Please provide your PAN number immediately to avoid penalties."
Thought Process:
1. Claims to be from a government authority, creating fear
2. Mentions penalties to create urgency
3. Asks for sensitive information (PAN number)
4. No case number or official reference provided
5. No alternative official verification method offered
Classification: Scam
Confidence: 0.92
Reasoning: This is attempting identity theft. Government departments would send official notices first, provide reference numbers, and never ask for full PAN details over an unsolicited call. The pressure tactics are red flags.
Follow-up questions:
- "Can you provide your employee ID and department division?"
- "What is the specific case/reference number for this issue?"
- "Can I call back on the official Income Tax Department helpline to verify this matter?"

### Example 3: Legitimate Call
Transcript: "Hello, I'm calling from ABC Telecom about the network upgrade in your area. You might experience brief service interruption tomorrow between 2 to 4 PM. No action required from your side."
Thought Process:
1. Provides specific company name and reason for calling
2. Only sharing information, not requesting any personal data
3. No urgency or pressure tactics
4. Gives specific timeframe
5. Explicitly states no action required
Classification: Not Scam
Confidence: 0.88
Reasoning: This appears to be a legitimate service notification from a telecom provider. It doesn't ask for any personal information, just informs about planned maintenance, and doesn't use high-pressure tactics.

When analyzing the new transcript, follow this chain of thought approach:
1. Look for common scam indicators (requests for personal/financial information, urgency, threats)
2. Analyze the legitimacy of the caller's claimed identity
3. Consider the nature of the request being made
4. Assess the pressure tactics or manipulation techniques
5. Draw conclusions based on comparison with known patterns

You MUST respond using the function `classify_scam_call_schema` with the correct JSON structure.
Be precise and helpful.
"""

SYSTEM_PROMPT_TEXT_FEW_SHOT = """
You are an AI Fraud/Scam Text Message Detector.

You will be given a transcript of a text message that may potentially be a scam.
Your task is to analyze this text and determine if it represents a scam message.

I'll provide you with several examples to help you understand different types of scam patterns. For each example, I'll show the thought process for classification.

## EXAMPLES OF TEXTS AND THOUGHT PROCESSES:

### Example 1: Financial Fraud
Text: "Warning: Suspicious login attempt on your bank account. Verify identity at http://trustbank-check.com now!"
Thought Process:
1. Creates urgency with "Warning" and "Suspicious login"
2. Directs to a suspicious URL (not an official bank domain)
3. Uses command language "Verify identity... now!"
4. No personalization or account details mentioned
5. Suspicious low-quality domain name with hyphens
Classification: Scam
Confidence: 0.98
Reasoning: This is a phishing attempt using fear tactics to make the recipient click a malicious link. Legitimate banks would never send such messages with suspicious domains and would typically reference the actual account or name of the customer.

### Example 2: Emotional Manipulation
Text: "Hi Dad, I'm in a crisis! Got arrested by mistake. Need Rs. 8,000 for bail. Pay now at http://urgentbail-help.com. Please save me!"
Thought Process:
1. Appeals to emotion with "crisis" and "save me"
2. Creates extreme urgency
3. Contains specific financial request
4. Uses suspicious URL for payment
5. Vague on details about the "arrest"
6. Uses manipulative language
Classification: Scam
Confidence: 0.96
Reasoning: This is an emotional manipulation scam targeting parents. Real emergencies involving legal issues wouldn't be resolved through random websites. The combination of emotional appeal, urgency, and suspicious payment method are clear red flags.

### Example 3: Normal Social Message
Text: "Hey, loved your latest TikTok vid! So funny!"
Thought Process:
1. Casual conversational tone
2. References specific content the recipient created
3. No requests for money or information
4. No links or suspicious elements
5. Natural messaging pattern between friends
Classification: Not Scam
Confidence: 0.97
Reasoning: This is clearly a normal social interaction referencing TikTok content the recipient posted. There are no requests, links, or attempts to manipulate the recipient.

### Example 4: Suspicious But Not Confirmed
Text: "Hi, it's your friend Neha. I'm struggling with bills and need 10,000 INR."
Thought Process:
1. Claims to be a friend but using a somewhat formal introduction
2. Directly asks for a significant amount of money
3. Provides a reason but with minimal details
4. No threatening language or extreme urgency
5. No suspicious links or payment methods mentioned
Classification: Suspicious
Confidence: 0.75
Reasoning: While this could be legitimate if Neha is actually a friend in need, the direct request for a substantial amount of money without much context is concerning. This pattern matches how scammers impersonate friends, but without more context, we can't be certain.

When analyzing the new text, follow this chain of thought approach:
1. Look for common scam indicators (suspicious URLs, requests for money, urgency)
2. Analyze the sender's claimed identity
3. Consider the nature of the request being made
4. Assess emotional manipulation techniques
5. Draw conclusions based on comparison with known patterns

You MUST respond using the function `classify_scam_text_schema` with the correct JSON structure.
Be precise and helpful.
"""

# --------------------------
# Retrieval Setup
# --------------------------
def get_retriever(top_k: int = 5, mode: str = "call"):
    if mode == "call":
        collection_name = CALL_QDRANT_COLLECTION_NAME
    elif mode == "text":
        collection_name = TEXT_QDRANT_COLLECTION_NAME
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs=HF_MODEL_KWARGS,
        encode_kwargs={"normalize_embeddings": True},
    )
    qdrant_client = QdrantClient(url=QDRANT_URL)

    vectorstore = QdrantVectorStore(
        embedding=embeddings,
        client=qdrant_client,
        collection_name=collection_name,
        vector_name="dense",
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

def get_llm(schema=None):
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        temperature=0.3,
        functions=[schema] if schema else [],
    )


def initialize_rag_components(top_k=5, mode: str = "call"):
    if mode == "call":
        retriever = get_retriever(top_k=top_k, mode=mode)
        llm = get_llm(schema=classify_scam_call_schema)
    elif mode == "text":
        retriever = get_retriever(top_k=top_k, mode=mode)
        llm = get_llm(schema=classify_scam_text_schema)
    return retriever, llm


# --------------------------
# RAG Runner with Function Calling
# --------------------------
def run_rag_query(query: str, retriever, llm, mode="call"):
    # Choose the appropriate system prompt based on mode
    if mode == "call":
        system_prompt = SYSTEM_PROMPT_CALL_RAG
    else:
        system_prompt = SYSTEM_PROMPT_TEXT_RAG
        
    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_template = HumanMessagePromptTemplate.from_template(
        "User query: {query}\n\nRetrieved documents:\n{context}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

    # Step 1: Retrieve context
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(
        f"Text: {doc.page_content}\nMetadata: {doc.metadata}" for doc in docs
    )

    raw_output = (chat_prompt | llm).invoke({
    "query": query,
    "context": context_text
    })
    print("RAW LLM OUTPUT:", raw_output)
    function_call = raw_output.additional_kwargs.get("function_call", {})
    arguments_str = function_call.get("arguments", "{}")
    parsed = json.loads(arguments_str)

    # 3. package up the response
    return {
        "results": [
            {"content": d.page_content, "metadata": d.metadata, "score": getattr(d, "score", None)}
            for d in docs
        ],
        "answer": parsed,
    }

# --------------------------
# Few-Shot Learning Implementation
# --------------------------
def run_few_shot_query(query: str, llm, mode="call", csv_path=None):
    """
    Run the scam detection using few-shot prompting instead of RAG.
    
    Args:
        query: The text or call transcript to analyze
        llm: The language model to use
        mode: Either "call" or "text" to determine the prompt and schema to use
        csv_path: Optional path to the dataset file (defaults to an appropriate path based on mode)
    
    Returns:
        Dict with classification results
    """
    # Choose the appropriate system prompt based on mode
    if mode == "call":
        system_prompt = SYSTEM_PROMPT_CALL_FEW_SHOT
    else:
        system_prompt = SYSTEM_PROMPT_TEXT_FEW_SHOT
    
    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_template = HumanMessagePromptTemplate.from_template(
        "Please analyze this transcript: {query}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
    
    raw_output = (chat_prompt | llm).invoke({
        "query": query
    })
    
    print("RAW LLM OUTPUT:", raw_output)
    function_call = raw_output.additional_kwargs.get("function_call", {})
    arguments_str = function_call.get("arguments", "{}")
    parsed = json.loads(arguments_str)
    
    # Return results without RAG context
    return {
        "results": [],  # No retrieval results in few-shot mode
        "answer": parsed,
    }

def get_llm_for_few_shot(mode="call"):
    """Get LLM configured with the appropriate function schema for few-shot learning"""
    if mode == "call":
        schema = classify_scam_call_schema
    else:
        schema = classify_scam_text_schema
    
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        temperature=0.3,
        functions=[schema]
    )


# if __name__ == "__main__":
#     query = "Sir, I’m calling from your bank. Please read out the OTP to verify your account."
#     top_k = 5
#     result = run_rag_pipeline(query, top_k)
#     print("\n=== Running Scam Classification ===")
#     response = run_rag_pipeline(query=query, top_k=top_k)
#     print("\n--- Retrieved Context Chunks ---")
#     for i, doc in enumerate(response["results"], 1):
#         print(f"\n[Doc {i}]")
#         print(f"Score: {doc['score']}")
#         print(f"Text: {doc['content']}")
#         print(f"Metadata: {doc['metadata']}")
#     print("\n--- Function Call Result ---")
#     print(response["answer"])