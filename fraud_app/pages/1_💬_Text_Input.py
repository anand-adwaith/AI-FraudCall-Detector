import streamlit as st
from utils import analyze_text_with_llm

st.title("💬 Text Message Scam Detector")

user_input = st.text_area("Enter message or call transcript:")
if st.button("🔍 Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = analyze_text_with_llm(user_input.strip())
        st.success(f"🔎 Prediction: **{result}**")
    else:
        st.warning("Enter some text.")
