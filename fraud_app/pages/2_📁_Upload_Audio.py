import streamlit as st
import whisper
from utils import analyze_text_with_llm

st.title("📁 Upload Audio File")

audio_file = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])
if audio_file:
    st.audio(audio_file)
    if st.button("🎙️ Transcribe & Analyze"):
        with st.spinner("Transcribing..."):
            model = whisper.load_model("base")
            with open("temp.wav", "wb") as f:
                f.write(audio_file.read())
            result = model.transcribe("temp.wav")
            transcript = result["text"]
        st.markdown(f"📝 Transcribed Text:\n> {transcript}")
        with st.spinner("Analyzing..."):
            prediction = analyze_text_with_llm(transcript.strip())
        st.success(f"🔎 Prediction: **{prediction}**")
