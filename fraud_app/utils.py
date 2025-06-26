
import requests

API_URL = "http://localhost:8000/api" 
def analyze_text_with_llm(text: str, mode, message_type="call"):
    """Send input to FastAPI analyze endpoint and return prediction result."""
    payload = {
        "query": text,
        "mode": mode,
        "message_type": message_type
    }

    try:
        response = requests.post(f"{API_URL}/analyze", json=payload)
        response.raise_for_status() 
        return response.json()  
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def analyze_audio(filepath,mode,lang):
    payload = {
        "file_path": filepath,       
        "language_id": lang,                    
        "model_type": "call",             
        "analysis_type": mode           

    }

    response = requests.post(f"{API_URL}/audio-classify", json=payload)

    if response.ok:
        return response.json()
    else:
        print("❌ Error:", response.status_code, response.text)
       

#print(analyze_text_with_llm('fdsafdsfdsfdfdf', 'rag', "text")['answer']['classification'])

