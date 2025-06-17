# utils.py
import requests

def analyze_text_with_llm(text: str) -> str:
    try:
        url = f"https://api.agify.io/?name={text}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection failed: {e}"
    

print(analyze_text_with_llm("John Doe") ) # Example usage, replace with actual text input in your application