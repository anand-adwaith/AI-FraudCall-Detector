import streamlit as st
import sounddevice as sd
import tempfile
import os
import uuid
from datetime import datetime
import scipy.io.wavfile
from utils import analyze_audio

st.title("🎤 Live Mic Recording")

def record_audio(duration=15, samplerate=16000):
    st.info("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio

UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mode = st.selectbox("Choose analysis mode", ["rag", "few_shot"])
lang = st.selectbox("Choose Audio Language", ["hi", "ta"])

if st.button("🎙️ Start 15-sec Recording"):
    audio = record_audio()
    

    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the audio file locally
    scipy.io.wavfile.write(file_path, 16000, audio)

    st.audio(file_path)
    #st.info(f"Audio saved to: {os.path.abspath(file_path)}")

    with st.spinner("Analyzing the audio file..."):
        try:
            # Pass full path to analysis function
            full_path = os.path.abspath(file_path)
            prediction = analyze_audio(full_path,mode,lang)

            # You can now print parts of the result if it's a dict
            if isinstance(prediction, dict):
                transcript = prediction['transcription']
                st.markdown(f"📝 Transcribed Text:\n> {transcript}")
                st.success(f"🔎 Prediction: **{prediction['classification']['classification']}**")
                
                with st.expander("🧠 Reasoning"):
                    st.write(prediction['classification']['reasoning'])
                with st.expander("❓ Follow-up Questions"):
                    questions = prediction['classification']['follow_up_questions']
                    if questions:
                        for q in questions:
                            st.markdown(f"- {q}")
                    else:
                        st.write("No follow-up questions available.")
            else:
                st.success(f"🔎 Prediction: {prediction}")

        except Exception as e:
            st.error(f"❌ Analysis Error: {e}")
