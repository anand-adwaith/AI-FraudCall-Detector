# utils.py
import requests

def analyze_text_with_llm(text: str) -> str:
    try:
        response = requests.post("http://localhost:8000/predict", json={"text": text})
        if response.status_code == 200:
            return response.json().get("prediction", "Unknown")
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection failed: {e}"
