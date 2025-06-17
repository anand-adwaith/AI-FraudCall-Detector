import streamlit as st
from utils import analyze_text_with_llm

# Set page configuration
st.set_page_config(page_title="Text Message Scam Detector", page_icon="💬", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat header
st.title("Text Message Scam Detector")

# Display chat messages using Streamlit's chat components
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from the LLM
    with st.chat_message("assistant"):
        response = analyze_text_with_llm(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})