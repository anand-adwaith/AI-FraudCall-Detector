from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16)



system_prompt = '''
You are an AI Fraud/Scam Call Detector.

You will be given a transcript of what a caller (potential scammer) said during a phone call.
Your task is to analyze this transcript and determine if it represents a scam call.

I'll provide you with several examples to help you understand different types of scam patterns. For each example, I'll show the thought process for 
classification.

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

Format your response like this:
                {
                  "classification": "Scam", "Not Scam", or "Suspicious",
                  "confidence": 0.95,
                  "reasoning": "Your detailed reasoning",
                  "follow_up_questions": ["Question 1?", "Question 2?", "Question 3?"]
                }

Please analyze this transcript:
Hello, Im bank manager of Canara bank, we noticed some unusual activity in your account, kindly share your otp to verify the transaction.
'''
system_prompt_v2 = "You are an AI Fraud/Scam Call Detector. Given user input, classify the call as scam or not scam, give one word answer, the user query is : Hello, Im bank manager of Canara bank, we noticed some unusual activity in your account, kindly share your otp to verify the transaction.."
system_prompt_v1 = "You are an AI Fraud/Scam Call Detector., given user input, classify the call as scam or not scam, provide confidence score and reasoning, and suggest follow-up questions if suspicious."

user_prompt = "Classify the following text with the help of system prompt instructions: "
messages = [
{"role": "user", "content": system_prompt_v2},
]

outputs = pipe(
messages,
max_new_tokens=1024
)
print(outputs[0])