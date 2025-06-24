import requests
import os
import random

# Sarvam AI API endpoints
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"


def translate(api_key, text, source, target):
    headers = {"api-subscription-key": api_key}

    payload = {
        "input": text,
        "source_language_code": source,
        "target_language_code": target,
        "enable_preprocessing": "true"
    }

    response = requests.post(SARVAM_TRANSLATE_URL,
                             json=payload,
                             headers=headers)
    response_data = response.json()
    translated_text = response_data.get("translated_text", "")
    return translated_text


def stt(api_key, audio_file_path):
    headers = {"api-subscription-key": api_key}

    audio_file = open(audio_file_path, "rb")
    if not audio_file:
        return '', ''

    filename = os.path.basename(audio_file_path)
    files = {'file': (filename, audio_file, 'audio/wav')}

    response = requests.post(SARVAM_STT_URL, headers=headers, files=files)
    result = response.json()
    transcript = result.get('transcript', '')
    language_code = result.get('language_code', '')

    return transcript, language_code


def chat(api_key, query, model, temperature):
    headers = {"api-subscription-key": api_key}

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": query
        }],
        "temperature": temperature
    }

    response = requests.post(SARVAM_CHAT_URL, headers=headers, json=payload)

    return response.json()


def tts(api_key, text, language, gender):
    speakers = {
        "female": ["anushka", "manisha", "vidya", "arya"],
        "male": ["abhilash", "karun", "hitesh"]
    }

    headers = {"api-subscription-key": api_key}

    # select random speaker with gender
    speaker = random.choice(speakers[gender])

    payload = {
        "text": text,
        "target_language_code": language,
        "speaker": speaker,
        "enable_preprocessing": "true",
        "pitch": 0,
        "pace": 1,
        "loudness": 1
    }

    response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers)
    response_data = response.json()

    return response_data.get("audios")
