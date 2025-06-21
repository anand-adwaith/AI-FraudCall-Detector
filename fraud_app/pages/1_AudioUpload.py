import streamlit as st
import os
import uuid
from datetime import datetime
from utils import analyze_audio

# Set page title
st.title("📁 Upload Audio File")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# File uploader for single file
audio_file = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])

mode = st.selectbox("Choose analysis mode", ["rag", "few_shot"])
lang = st.selectbox("Choose Audio Language", ["hi", "ta"])
if audio_file:
    # Display the audio player
    st.audio(audio_file)
    
    # Transcribe & Analyze button
    if st.button("🎙️ Transcribe & Analyze"):
        # Generate unique filename
        file_extension = os.path.splitext(audio_file.name)[1]
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file locally
        with open(file_path, "wb") as f:
            f.write(audio_file.getvalue())
        
        st.info(f"File saved as: {unique_filename}")
        
        full_path = os.path.abspath(file_path)
        # Call your API with the file path
        with st.spinner("Calling transcription API..."):

            try:

                
                # Analyze the transcript
                with st.spinner("Analyzing..."):
                    prediction = analyze_audio(full_path,mode,lang)
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
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Clean up file on error
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Add option to delete the file after processing
        if st.button("🗑️ Delete File"):
            if os.path.exists(file_path):
                os.remove(file_path)
                st.info(f"File {unique_filename} deleted.")