import streamlit as st

st.set_page_config(page_title="Fraud Detector Home", layout="centered")
st.title("🛡️ Welcome to the Fraud Call/Text Detector App")
st.markdown("""
This app helps identify scam messages or calls using LLMs and ASR.  
Navigate through tabs on the left for:
- 💬 Text Input  
- 📁 Audio Upload  
- 🎤 Live Recording  

> Powered by GPT and Whisper.
""")
