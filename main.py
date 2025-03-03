import streamlit as st
import asyncio
from langchain_openai.chat_models import ChatOpenAI

st.title("Web Content Q&A Tool")

# Sidebar settings
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "prev_key" not in st.session_state:
    st.session_state.prev_key = ""
if "model" not in st.session_state:
    st.session_state.model = None  # Ensure model is initialized

@st.cache_resource
def get_model(api_key, temperature):
    return ChatOpenAI(temperature=temperature, api_key=api_key)


async def chat_stream(model, prompt):
    try:
        response = await asyncio.to_thread(model.invoke, prompt)
        for char in response:
            yield char
            await asyncio.sleep(0.02)
    except Exception as e:
        yield f"Exception occurred: {e}"

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

# Ensure model is defined before use
if openai_api_key.startswith("sk-"):
    if st.session_state.prev_key != openai_api_key:
        # print("Here")
        st.session_state.prev_key = openai_api_key
        st.session_state.model = get_model(openai_api_key, temperature)

# Display chat history
for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            st.session_state[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )

if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
elif st.session_state.model:  # Ensure model is available before proceeding
    if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response_text = st.write_stream(chat_stream(st.session_state.model, prompt))
            st.session_state.history.append({"role": "assistant", "content": response_text})

            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                on_change=save_feedback,
                args=[len(st.session_state.history)],
            )
