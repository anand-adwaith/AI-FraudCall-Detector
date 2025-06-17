import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile
import requests
from utils import analyze_text_with_llm

st.title("🎤 Live Mic Recording")

def record_audio(duration=5, samplerate=16000):
    st.info("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio

def send_to_sarvam_api(path):
    API_URL = "https://your-sarvam-api-url.com/transcribe"
    with open(path, "rb") as f:
        response = requests.post(API_URL, files={"file": f})
    return response.json()

if st.button("🎙️ Start 5-sec Recording"):
    audio = record_audio()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, 16000, audio)
        wav_path = tmpfile.name
    st.audio(wav_path)

    with st.spinner("Transcribing via Sarvam..."):
        try:
            result = send_to_sarvam_api(wav_path)
            transcript = result.get("text", "")
            if not transcript:
                st.error("❌ No transcription received.")
            else:
                st.markdown(f"📝 Transcribed Text:\n> {transcript}")
                with st.spinner("Analyzing..."):
                    prediction = analyze_text_with_llm(transcript.strip())
                st.success(f"🔎 Prediction: **{prediction}**")
        except Exception as e:
            st.error(f"❌ API Error: {e}")
