import streamlit as st
import os
import uuid
from datetime import datetime
from utils import analyze_text_with_llm

# Set page title
st.title("📁 Upload Audio File")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# File uploader for single file
audio_file = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])

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
        
        # Call your API with the file path
        with st.spinner("Calling transcription API..."):
            # Replace this with your actual API call
            # For now, let's simulate it with a direct file read
            try:
                # This is where you would make your API call
                # Example: response = requests.post(API_URL, files={'file': open(file_path, 'rb')})
                # For demonstration, we'll just use the file path
                
                # REPLACE THIS with your actual API call
                transcript = f"This is a simulated transcript for file: {unique_filename}"
                
                # Display the transcript
                st.markdown(f"📝 Transcribed Text:\n> {transcript}")
                
                # Analyze the transcript
                with st.spinner("Analyzing..."):
                    prediction = analyze_text_with_llm(transcript.strip())
                st.success(f"🔎 Prediction: **{prediction}**")
                
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