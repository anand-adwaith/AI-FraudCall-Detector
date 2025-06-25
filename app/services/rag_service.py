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
from transformers import pipeline
import logging

# Configure logging
logger = logging.getLogger("rag-service")
# find .env in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.tools.convert_to_openai import format_tool_to_openai_function

from app.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
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
    collection_name = QDRANT_COLLECTION_NAME
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
def run_rag_query(query: str, retriever, llm, mode="call", model_type="GPT"):
    # Choose the appropriate system prompt based on mode
    if mode == "call":
        system_prompt = SYSTEM_PROMPT_CALL_RAG
        schema = classify_scam_call_schema
    else:
        system_prompt = SYSTEM_PROMPT_TEXT_RAG
        schema = classify_scam_text_schema
        
    # Prepare retrieved documents
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    # Create a result list with retriever output
    results = []
    for i, doc in enumerate(docs):
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": getattr(doc, "score", None)
        })
      # Handle different model types
    if model_type.lower() == "gemma":
        # Use Gemma model for response generation
        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nRetrieved documents:\n{context}"
        parsed = generate_with_gemma(full_prompt, schema)
        
        # Return the same structure as GPT version
        return {
            "results": results,
            "answer": parsed
        }
    elif model_type.lower() == "llama":
        # Use Llama model for response generation
        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nRetrieved documents:\n{context}"
        parsed = generate_with_llama(full_prompt, schema)
        
        # Return the same structure as GPT version
        return {
            "results": results,
            "answer": parsed
        }
    else:
        # Default GPT approach
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
    #print("RAW LLM OUTPUT:", raw_output)
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
def run_few_shot_query(query: str, llm, mode="call", csv_path=None, model_type="GPT"):
    """
    Run the scam detection using few-shot prompting instead of RAG.
    
    Args:
        query: The text or call transcript to analyze
        llm: The language model to use
        mode: Either "call" or "text" to determine the prompt and schema to use
        csv_path: Optional path to the dataset file (defaults to an appropriate path based on mode)
        model_type: Model to use for inference - "GPT" or "gemma"
    
    Returns:
        Dict with classification results
    """
    # Choose the appropriate system prompt based on mode
    if mode == "call":
        system_prompt = SYSTEM_PROMPT_CALL_FEW_SHOT
        schema = classify_scam_call_schema
    else:
        system_prompt = SYSTEM_PROMPT_TEXT_FEW_SHOT
        schema = classify_scam_text_schema
      # Handle different model types
    if model_type.lower() == "gemma":
        # Use Gemma model for response generation
        full_prompt = f"{system_prompt}\n\nPlease analyze this transcript: {query}"
        logger.info(f"Full Prompt generated for Gemma: {full_prompt[:100]}...")  # Log just the beginning to avoid excessive logs
        parsed = generate_with_gemma(full_prompt, schema)
        
        # Return the same structure as GPT version
        return {
            "results": [],  # No retrieval results in few-shot mode
            "answer": parsed
        }
    elif model_type.lower() == "llama":
        # Use Llama model for response generation
        full_prompt = f"{system_prompt}\n\nPlease analyze this transcript: {query}"
        logger.info(f"Full Prompt generated for Llama: {full_prompt[:100]}...")  # Log just the beginning to avoid excessive logs
        parsed = generate_with_llama(full_prompt, schema)
        
        # Return the same structure as GPT version
        return {
            "results": [],  # No retrieval results in few-shot mode
            "answer": parsed
        }
    else:
        # Default GPT approach
        system_template = SystemMessagePromptTemplate.from_template(system_prompt)
        human_template = HumanMessagePromptTemplate.from_template(
            "Please analyze this transcript: {query}"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
        
        raw_output = (chat_prompt | llm).invoke({
            "query": query
        })
        
        #print("RAW LLM OUTPUT:", raw_output)
        function_call = raw_output.additional_kwargs.get("function_call", {})
        arguments_str = function_call.get("arguments", "{}")
        parsed = json.loads(arguments_str)
        
        # Return results without RAG context
        return {
            "results": [],  # No retrieval results in few-shot mode
            "answer": parsed,
        }

# Get HuggingFace model
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

# Function to generate structured responses with Gemma
def generate_with_gemma(prompt: str, schema: dict = None, max_tokens: int = 1024):
    """
    Generate a structured response using the Gemma model
    
    Args:
        prompt: The prompt to send to the model
        schema: The schema for structuring the response
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing the structured response
    """
    import json
    import torch
    import re
    from app.utils import ModelManager
    
    try:
        # Get Gemma model and tokenizer
        model, tokenizer = ModelManager.get_gemma_model_and_tokenizer()
        logger.info("Retrieved Gemma model and tokenizer successfully")
        
        # Format the prompt to include schema instructions if provided
        if schema:
            # Extract expected output format from schema
            output_format = {
                key: schema["parameters"]["properties"][key]["description"]
                for key in schema["parameters"]["required"]
            }
            
            # Add optional properties if available
            if "follow_up_questions" in schema["parameters"]["properties"]:
                output_format["follow_up_questions"] = schema["parameters"]["properties"]["follow_up_questions"]["description"]
              # Build JSON format instruction with explicit examples to avoid confusion
            json_format_instruction = """Your response MUST be in valid JSON format with these fields:
            ```json
            {
              "classification": "Scam", 
              "confidence": 0.95,
              "reasoning": "Detailed explanation of why this is classified as a scam."
            }
            ```
            
            For suspicious cases, also include follow-up questions:
            ```json
            {
              "classification": "Suspicious",
              "confidence": 0.75,
              "reasoning": "Explanation of why this is suspicious",
              "follow_up_questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?"
              ]
            }
            ```"""
            
            # Create messages in the format suitable for apply_chat_template
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": json_format_instruction}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    },
                ]
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
        else:
            # When no schema is provided, use a simpler message structure
            messages = [
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    },
                ]
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            
        logger.info("loaded inputs for Gemma model successfully")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,  # Match GPT temperature
                top_p=0.95,
                repetition_penalty=1.2
            )
            
        logger.info("Decoding Gemma model outputs...")
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
          # Print sample for debugging
        #logger.info(f"Gemma response (first 200 chars): {full_response[:200]}")
        #logger.debug(f"Full raw response: {full_response}")

        # Try to extract JSON using the specialized function
        parsed = extract_json_from_gemma_response(full_response)
        if parsed:
            logger.info("Successfully extracted JSON using specialized function")
            return parsed
        
        # If specialized extraction fails, fallback to manual parsing (less preferred)
        logger.warning("Specialized JSON extraction failed, falling back to manual parsing")
          # Use the specialized JSON extraction function
        parsed_json = extract_json_from_gemma_response(full_response)
        
        if parsed_json:
            logger.info("Successfully extracted JSON using specialized extractor")
            
            # Fix confidence type if it's a string
            if "confidence" in parsed_json and isinstance(parsed_json["confidence"], str):
                try:
                    parsed_json["confidence"] = float(parsed_json["confidence"])
                except ValueError:
                    parsed_json["confidence"] = 0.5  # Default fallback
            
            # Ensure follow_up_questions is a list if present
            if "follow_up_questions" in parsed_json and not isinstance(parsed_json["follow_up_questions"], list):
                if isinstance(parsed_json["follow_up_questions"], str):
                    # If it's the schema description, create default questions
                    if parsed_json["follow_up_questions"] == "Follow-up questions if Suspicious":
                        parsed_json["follow_up_questions"] = [
                            "What is the purpose of sharing the OTP?", 
                            "Can you verify your identity and organization?", 
                            "Is there an official channel I should use instead?"
                        ]
                    else:
                        # Convert single question to list
                        parsed_json["follow_up_questions"] = [parsed_json["follow_up_questions"]]
            
            # Verify required fields if schema provided
            if schema:
                required_keys = schema["parameters"]["required"]
                if all(key in parsed_json for key in required_keys):
                    return parsed_json
          # If code block extraction fails, look for JSON objects with curly braces
        try:
            json_pattern = re.compile(r'\{[^\}]*(?:\{[^\}]*\})[^\}]*\}|\{[^\}]*\}')
            matches = json_pattern.findall(full_response)
            if matches:
                # Try each match, starting with the longest (most complete) one
                matches.sort(key=len, reverse=True)
                for match in matches:
                    try:
                        parsed_json = json.loads(match)
                        
                        # Fix data types to match schema requirements
                        if "confidence" in parsed_json and isinstance(parsed_json["confidence"], str):
                            try:
                                parsed_json["confidence"] = float(parsed_json["confidence"])
                            except ValueError:
                                parsed_json["confidence"] = 0.5
                        
                        # Ensure follow_up_questions is a list if present
                        if "follow_up_questions" in parsed_json and not isinstance(parsed_json["follow_up_questions"], list):
                            if isinstance(parsed_json["follow_up_questions"], str):
                                # If it's just a string description, create a default list
                                if parsed_json["follow_up_questions"] == "Follow-up questions if Suspicious":
                                    parsed_json["follow_up_questions"] = [
                                        "What is the purpose of sharing the OTP?", 
                                        "Can you verify your identity and organization?", 
                                        "Is there an official channel I should use instead?"
                                    ]
                                else:
                                    # Convert single question to list
                                    parsed_json["follow_up_questions"] = [parsed_json["follow_up_questions"]]
                        
                        # Verify required fields if schema provided
                        if schema:
                            required_keys = schema["parameters"]["required"]
                            if all(key in parsed_json for key in required_keys):
                                logger.info("Successfully extracted JSON from text with regex")
                                return parsed_json
                    except json.JSONDecodeError:
                        continue
        except Exception as regex_error:
            logger.warning(f"Error in regex JSON extraction: {str(regex_error)}")
        
        # If JSON extraction fails, make a best-effort classification based on keywords
        logger.warning("JSON extraction failed, falling back to keyword analysis")
        response_text = full_response.lower()
        
        # Fallback parsing - extract classification
        if "not scam" in response_text:
            classification = "Not Scam"
        elif "scam" in response_text:
            classification = "Scam"
        elif "suspicious" in response_text:
            classification = "Suspicious"
        else:
            classification = "Unknown"
            
        # Try to extract confidence
        confidence_match = re.search(r'"confidence":\s*(0\.\d+|1\.0)', response_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Try to extract reasoning
        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response_text)
        reasoning = reasoning_match.group(1) if reasoning_match else f"Extracted from unstructured response. Response indicates potential {classification.lower()}."
          # Build response with proper data types
        result = {
            "classification": classification,
            "confidence": float(confidence),  # Ensure it's a float
            "reasoning": reasoning
        }
        
        # Add follow-up questions if applicable
        if schema and "follow_up_questions" in schema["parameters"]["properties"] and classification == "Suspicious":
            # Extract questions if possible
            questions_match = re.search(r'"follow_up_questions":\s*\[(.*?)\]', full_response, re.DOTALL)
            if questions_match:
                try:
                    questions_text = "[" + questions_match.group(1) + "]"
                    questions = json.loads(questions_text.replace("'", '"'))
                    result["follow_up_questions"] = questions
                except Exception as e:
                    logger.warning(f"Failed to parse follow-up questions JSON: {e}")
                    # Fallback: extract quoted strings that end with question marks
                    questions = re.findall(r'"([^"]+\?)"', full_response)
                    if questions:
                        result["follow_up_questions"] = questions[:3]  # Limit to 3 questions
                    else:
                        # Default questions if none found
                        result["follow_up_questions"] = [
                            "What is this OTP being used for specifically?", 
                            "Can you provide official verification of your identity?", 
                            "Is there an alternative way to verify this request?"
                        ]
            else:
                # Default questions if pattern not found
                result["follow_up_questions"] = [
                    "What is this OTP being used for specifically?", 
                    "Can you provide official verification of your identity?", 
                    "Is there an alternative way to verify this request?"
                ]
        
        return result
        
    except Exception as e:
        # Return error details with more context
        logger.error(f"Gemma model generation error: {str(e)}", exc_info=True)
        
        return {
            "classification": "Error",
            "confidence": 0.0,
            "reasoning": f"Error using Gemma model: {str(e)}. Please check server logs for details."
        }

def extract_json_from_gemma_response(full_response: str):
    """
    Extract JSON from Gemma model response with specialized handling for common patterns.
    
    Args:
        full_response: The full response from the Gemma model
    
    Returns:
        Parsed JSON object or None if extraction fails
    """
    import json
    import re
    
    # 1. Look for "model" keyword followed by JSON in code block - a common pattern
    model_json_match = re.search(r'model\s*```(?:json)?\s*([\s\S]*?)```', full_response)
    if model_json_match:
        try:
            json_str = model_json_match.group(1).strip()
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            pass
    
    # 2. Look for any JSON code blocks
    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', full_response)
    for block in json_blocks:
        try:
            parsed_json = json.loads(block.strip())
            return parsed_json
        except json.JSONDecodeError:
            continue
    
    # 3. Look for JSON objects with curly braces
    json_pattern = re.compile(r'\{[^\}]*(?:\{[^\}]*\})[^\}]*\}|\{[^\}]*\}')
    matches = json_pattern.findall(full_response)
    if matches:
        # Try each match, starting with the longest (most complete) one
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                parsed_json = json.loads(match)
                return parsed_json
            except json.JSONDecodeError:
                continue
    
    return None

# Function to generate structured responses with Llama
def generate_with_llama(prompt: str, schema: dict = None, max_tokens: int = 256):
    """
    Generate a structured response using the Llama 3.2 model
    
    Args:
        prompt: The prompt to send to the model
        schema: The schema for structuring the response
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary containing the structured response
    """
    import json
    import re
    from app.utils import ModelManager
    
    try:
        # Get Llama pipeline
        pipe = ModelManager.get_llama_pipeline()
        logger.info("Retrieved Llama pipeline successfully")
        
        # Format the input messages based on schema
        if schema:
            # Create a system message with instructions
            system_content = "You are an AI Fraud/Scam Detector. "
            system_content += "Your response MUST be in valid JSON format. "
            
            if "follow_up_questions" in schema["parameters"]["properties"]:
                system_content += """
                Format your response like this:
                {
                  "classification": "Scam", "Not Scam", or "Suspicious",
                  "confidence": 0.95,
                  "reasoning": "Your detailed reasoning",
                  "follow_up_questions": ["Question 1?", "Question 2?", "Question 3?"]
                }
                """
            else:
                system_content += """
                Format your response like this:
                {
                  "classification": "Scam", "Not Scam", or "Suspicious",
                  "confidence": 0.95,
                  "reasoning": "Your detailed reasoning"
                }
                """
                
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        logger.info("Generating response with Llama model")
        outputs = pipe(messages, max_new_tokens=max_tokens)
        
        # Extract the assistant's response
        llama_response = outputs[0]["generated_text"][-1]
          # Check the response format and extract content
        if isinstance(llama_response, dict) and "content" in llama_response:
            # Standard format from the model
            response_text = llama_response.get("content", "")
        else:
            # If the model returns the text directly
            logger.warning(f"Unexpected response format from Llama, trying to use response directly: {llama_response}")
            response_text = str(llama_response)
        logger.info(f"Llama response (first 100 chars): {response_text[:100]}")
        
        # Try to extract JSON from the response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                
                # Ensure proper types
                if "confidence" in parsed_json and isinstance(parsed_json["confidence"], str):
                    try:
                        parsed_json["confidence"] = float(parsed_json["confidence"])
                    except ValueError:
                        parsed_json["confidence"] = 0.5
                
                # Ensure follow_up_questions is a list if needed
                if schema and "follow_up_questions" in schema["parameters"]["properties"]:
                    if "follow_up_questions" not in parsed_json and parsed_json.get("classification") == "Suspicious":
                        parsed_json["follow_up_questions"] = [
                            "What is the purpose of the requested information?",
                            "Can you verify your identity?",
                            "Is there an official channel for this request?"
                        ]
                    elif "follow_up_questions" in parsed_json and not isinstance(parsed_json["follow_up_questions"], list):
                        parsed_json["follow_up_questions"] = [parsed_json["follow_up_questions"]]
                
                logger.info("Successfully extracted JSON response from Llama")
                return parsed_json
            else:
                logger.warning("No JSON found in Llama response")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Llama response: {e}")
        
        # Fallback to keyword-based response
        response_text_lower = response_text.lower()
        if "not scam" in response_text_lower:
            classification = "Not Scam"
        elif "scam" in response_text_lower:
            classification = "Scam"
        elif "suspicious" in response_text_lower:
            classification = "Suspicious"
        else:
            classification = "Unknown"
        
        result = {
            "classification": classification,
            "confidence": 0.5,  # Default confidence
            "reasoning": f"Extracted from unstructured response: {response_text[:150]}..."
        }
        
        # Add follow-up questions if applicable
        if schema and "follow_up_questions" in schema["parameters"]["properties"] and classification == "Suspicious":
            result["follow_up_questions"] = [
                "What is the purpose of the requested information?",
                "Can you verify your identity?",
                "Is there an official channel for this request?"
            ]
        
        return result
    
    except Exception as e:
        # Provide detailed error information
        logger.error(f"Llama model generation error: {str(e)}", exc_info=True)
        
        return {
            "classification": "Error",
            "confidence": 0.0,
            "reasoning": f"Error using Llama model: {str(e)}. Please check server logs for details."
        }
