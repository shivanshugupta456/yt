from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Mood Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 30px;
}

.mode-card {
    background-color: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🤖 AI Mood Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Choose your AI personality and enjoy chatting</div>',
    unsafe_allow_html=True
)

# ---------------- MODEL ----------------
model = ChatMistralAI(model="mistral-small-2506")

# ---------------- MODES ----------------
modes = {
    "😡 Angry Mode": "You are an angry ai agent very angry",
    "😂 Funny Mode": "You are a funny ai agent very funny",
    "😢 Sad Mode": "You are a sad ai agent very sad"
}

selected_mode = st.selectbox(
    "Choose AI Mode",
    list(modes.keys())
)

system_prompt = modes[selected_mode]

# ---------------- SESSION STATE ----------------
if "current_mode" not in st.session_state:
    st.session_state.current_mode = selected_mode

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Reset chat when mode changes
if st.session_state.current_mode != selected_mode:
    st.session_state.current_mode = selected_mode

    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

    st.session_state.chat_history = []

# ---------------- DISPLAY CHAT ----------------
for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# ---------------- USER INPUT ----------------
prompt = st.chat_input("Type your message here...")

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt
    })

    # Add Human Message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # AI Response
    response = model.invoke(st.session_state.messages)

    # Add AI Message
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response.content
    })