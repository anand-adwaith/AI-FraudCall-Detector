import streamlit as st
from utils import analyze_text_with_llm

# Set page configuration
st.set_page_config(page_title="Text Message Scam Detector", page_icon="💬", layout="centered")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "rag"  # Default mode

# UI: Mode selector
mode = st.selectbox("Choose analysis mode", ["rag", "few_shot"])

# Check if the mode changed
if mode != st.session_state.last_mode:
    st.session_state.messages = []  # Clear chat
    st.session_state.last_mode = mode  # Update mode in session

# Page title
st.title("Text Message Scam Detector")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = analyze_text_with_llm(prompt.strip(), mode, 'text')
        st.markdown(response['answer']['classification'])
        st.markdown(response['answer']['reasoning'])

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response['answer']['classification']})
