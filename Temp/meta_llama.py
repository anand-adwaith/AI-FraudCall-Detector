import torch
from transformers import pipeline
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token="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"
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=1024,
)
print(outputs[0]["generated_text"][-1])