# StRAG-Chatbot: AI Chatbot with Memory Checkpointing & Feedback

An interactive AI chatbot built with Streamlit that leverages LangChain, Google Generative AI, and LangGraph to provide a seamless conversational experience with memory checkpointing and integrated user feedback.

---

## Overview

This project presents an AI chatbot application that:
- Utilizes **Streamlit** for an intuitive web-based chat interface.
- Employs **LangChain** with Google Generative AI for generating conversational responses.
- Implements a state graph workflow with **LangGraph** to manage conversation states.
- Provides memory checkpointing using **MemorySaver** for session persistence.
- Enables user feedback (thumbs up/down) on assistant responses to improve future interactions.

---

## Features

- **Real-time Conversational Interface:** Enjoy a dynamic chat experience with asynchronous response streaming.
- **Memory Checkpointing:** Maintains conversation context across sessions using LangGraph's state management.
- **User Feedback Integration:** Rate responses with feedback buttons directly within the chat for ongoing improvement.
- **Configurable Parameters:** Adjust key settings such as temperature and maximum chat history length via the sidebar.
- **Google Generative AI Integration:** Powered by ChatGoogleGenerativeAI to deliver context-aware and helpful responses.

---

## Installation

### Prerequisites

- Python 3.8 or later
- [Streamlit](https://streamlit.io/)
- Required Python libraries:
  - `asyncio`
  - `streamlit`
  - `langchain_google_genai`
  - `langgraph`
  - `langchain_core`
  - `pydantic`
  
You can install the dependencies using pip. For example:

```bash
pip install streamlit langchain_google_genai langgraph pydantic
```

> **Note:** Ensure that all libraries are compatible with your Python version.

---

## Setup & Configuration

1. **Google API Key:**
   - The application requires a valid Google API key. 
   - When you start the app, enter your key in the sidebar. A simple validation ensures the key begins with "A".
   - Alternatively, you can set the `OPENAI_API_KEY` environment variable.

2. **Adjustable Settings:**
   - **Temperature:** Control the randomness of AI responses (slider between 0.0 and 1.0).
   - **Max Chat History Length:** Define how many past messages to retain in the session (between 5 and 50).

---

## Running the Application

Run the chatbot using Streamlit from your terminal:

```bash
streamlit run your_script.py
```

Replace `your_script.py` with the filename containing the code.

Once running, the Streamlit interface will load in your web browser. You can interact with the chat, view conversation history, and provide feedback on assistant responses.

---

## How It Works

### Chat History Manager

- **Message Storage:** All messages (user and assistant) are stored in session state, along with feedback.
- **Feedback Mechanism:** Assistant messages display "👍" and "👎" buttons. Feedback is recorded and displayed for each message.

### Conversation Service

- **StateGraph Workflow:** A state graph defines nodes for both user and assistant interactions. The conversation flows through nodes with memory checkpointing.
- **Asynchronous Streaming:** Responses are generated and streamed in chunks asynchronously to create a responsive chat experience.
- **Integration with Google Generative AI:** The chatbot leverages the ChatGoogleGenerativeAI model to generate context-aware responses based on conversation history.

### Streamlit UI

- **Sidebar Configuration:** Easily adjust settings (API key, temperature, chat history length) via the sidebar.
- **Chat Display:** Conversations are rendered in a chat-style layout. Each message appears in a chat bubble, with feedback options available for the assistant's responses.

---

## Extending the Application

- **Custom Workflows:** Modify the state graph or prompt templates to suit your conversational needs.
- **Persistent Memory:** Integrate external storage solutions if you require aggregated feedback or session persistence beyond in-memory storage.
- **Enhanced UI:** Customize the Streamlit interface further with additional components or styling.

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests. If you encounter issues or have suggestions for improvements, please open an issue on the project repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the intuitive web UI framework.
- [LangChain](https://python.langchain.com/) and [Google Generative AI](https://developers.google.com/ai) for powerful AI integrations.
- [LangGraph](https://github.com/langgraph/langgraph) for stateful conversation management.

Enjoy chatting and enhancing the experience with your valuable feedback!