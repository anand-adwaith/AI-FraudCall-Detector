from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch
HF_READ = "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"
HF_WRITE = "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"
#set the environment variables for Hugging Face authentication
import os
os.environ["HF_READ_TOKEN"] = HF_READ
os.environ["HF_WRITE_TOKEN"] = HF_WRITE
# Initialize the Hugging Face authentication
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

model_id = "google/gemma-3-1b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, token = HF_READ, device_map="cuda"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id,token = HF_READ,device_map="cuda")

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(torch.bfloat16)


with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64)

outputs = tokenizer.batch_decode(outputs)
print(outputs[0])