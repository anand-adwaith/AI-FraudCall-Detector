import streamlit as st
import openai
import os
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile
import requests

# Configure your OpenAI/Sarvam API key
openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Fraud Call Detector", layout="centered")

st.title("🕵️‍♂️ Fraud Call/Text Detector")
st.markdown("Use this tool to check if a **chat message or audio call** might be a scam using an LLM model.")

# ---- Few-Shot Prompt Template ----
FEW_SHOT_PROMPT = """
You are a scam detector. Based on the text, predict whether the message is a scam or not.

Examples:
Transcript: "Hello sir, your Paytm KYC is expired. Please share your Aadhaar number and OTP."
Prediction: Scam

Transcript: "Your Ola ride has been confirmed for 6:30 PM."
Prediction: Not a scam

Transcript: "Hi, I’ve lost my phone. Send me money urgently."
Prediction: Scam

Transcript: "Hey Ramesh, are you joining the office call at 3 PM?"
Prediction: Not a scam

Now classify the following:
Transcript: "{input_text}"
Prediction:
"""

# ---- Chat Section ----
st.subheader("💬 Text Input")
user_input = st.text_area("Enter the text from the call or message:", height=150)

if st.button("🔍 Analyze Text"):
    if user_input.strip():
        prompt = FEW_SHOT_PROMPT.format(input_text=user_input.strip())
        with st.spinner("Analyzing with LLM..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            prediction = response.choices[0].message.content.strip()
            st.success(f"🔎 LLM Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text before clicking Analyze.")

# ---- Audio Upload Section ----
st.subheader("🎧 Upload Call Recording (optional)")
audio_file = st.file_uploader("Upload audio file (MP3/WAV)", type=["mp3", "wav"])
if audio_file:
    st.audio(audio_file)
    if st.button("🎙️ Transcribe & Analyze Audio"):
        with st.spinner("Transcribing..."):
            import whisper
            model = whisper.load_model("base")
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.read())
            result = model.transcribe("temp_audio.wav")
            transcript = result["text"]
            st.markdown(f"📝 Transcribed Text:\n> {transcript}")
            prompt = FEW_SHOT_PROMPT.format(input_text=transcript.strip())
            with st.spinner("Analyzing transcript with LLM..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                prediction = response.choices[0].message.content.strip()
                st.success(f"🔎 LLM Prediction: **{prediction}**")

# ---- Live Mic Streaming Section ----
st.subheader("🎤 Live Microphone Audio (Real-Time Simulation)")

def record_audio_chunk(duration=5, samplerate=16000):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio

def send_audio_to_api(wav_path):
    # 🔁 Replace this URL with your actual Sarvam endpoint
    API_ENDPOINT = "https://your-sarvam-api-url.com/transcribe"
    with open(wav_path, "rb") as audio_file:
        response = requests.post(API_ENDPOINT, files={"file": audio_file})
    return response.json()  # Adjust based on your API’s response format

if st.button("🎙️ Start Live Recording & Analyze"):
    audio = record_audio_chunk(duration=5, samplerate=16000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav_path = tmpfile.name
        scipy.io.wavfile.write(wav_path, 16000, audio)
    st.audio(wav_path)

    with st.spinner("Sending to Sarvam for transcription..."):
        try:
            result = send_audio_to_api(wav_path)
            transcript = result.get("text", "")
            if not transcript:
                st.error("❌ No transcription received from Sarvam.")
            else:
                st.markdown(f"📝 Transcribed Text:\n> {transcript}")
                prompt = FEW_SHOT_PROMPT.format(input_text=transcript.strip())
                with st.spinner("Analyzing with LLM..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    )
                    prediction = response.choices[0].message.content.strip()
                    st.success(f"🔎 LLM Prediction: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Error sending audio to Sarvam: {e}")

