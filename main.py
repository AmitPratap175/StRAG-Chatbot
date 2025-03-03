import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import List, Dict, Generator, TypedDict

# Configuration
class AppConfig(BaseModel):
    google_api_key: str = Field(..., env="OPENAI_API_KEY")
    temperature: float = 0.5
    max_history_length: int = 10

# State definition for the LangGraph workflow
class ChatState(TypedDict):
    messages: List[Dict[str, str]]

# Chat history manager
class ChatHistoryManager:
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        if "history" not in st.session_state:
            st.session_state.history = []
        # For demonstration, feedback is stored with each message in the history.
        # In production, for aggregated feedback across sessions, use an external store.

    def add_message(self, role: str, content: str):
        # Append the message with an optional feedback field (None initially)
        st.session_state.history.append({"role": role, "content": content, "feedback": None})
        if len(st.session_state.history) > self.max_length:
            st.session_state.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        return st.session_state.history

    def update_feedback(self, index: int, feedback: str):
        try:
            st.session_state.history[index]["feedback"] = feedback
        except IndexError:
            pass

    def clear_history(self):
        st.session_state.history = []

# LangGraph conversation service
class ConversationService:
    def __init__(self, llm: ChatGoogleGenerativeAI, history_manager: ChatHistoryManager):
        self.llm = llm
        self.history_manager = history_manager
        self.checkpointer = MemorySaver()
        
        # Define the workflow
        workflow = StateGraph(ChatState)
        
        # Define nodes
        workflow.add_node("assistant", self.generate_response)
        workflow.add_node("user", self.passthrough)
        
        # Define edges
        workflow.add_edge("user", "assistant")
        workflow.add_edge("assistant", END)
        
        workflow.set_entry_point("user")
        
        # Compile with checkpointing
        self.app = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_after=["assistant"]
        )

    async def generate_response(self, state: ChatState) -> ChatState:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"messages": state["messages"]})
        
        return {"messages": state["messages"] + [{"role": "assistant", "content": response.content}]}

    async def passthrough(self, state: ChatState) -> ChatState:
        return state

    async def generate_stream(self, input_text: str, thread_id: str = "default") -> Generator[str, None, None]:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Build the state using the entire conversation history plus the new user message.
            new_state = {
                "messages": self.history_manager.get_history() + [{"role": "user", "content": input_text}]
            }
            
            async for event in self.app.astream(
                new_state,
                config=config,
                stream_mode="values"
            ):
                if event["messages"][-1]["role"] == "assistant":
                    yield event["messages"][-1]["content"]
                    await asyncio.sleep(0.02)
                else:
                    yield ""
                    await asyncio.sleep(0.02)
                    
        except Exception as e:
            yield f"Error: {str(e)}"

# Streamlit UI
class ChatUI:
    @staticmethod
    def setup_sidebar() -> AppConfig:
        st.sidebar.title("Settings")
        return AppConfig(
            google_api_key=st.sidebar.text_input("Google API Key", type="password"),
            temperature=st.sidebar.slider("Temperature", 0.0, 1.0, 0.5),
            max_history_length=st.sidebar.number_input("Max Chat History Length", min_value=5, max_value=50, value=10)
        )

    @staticmethod
    def display_chat(history_manager: ChatHistoryManager):
        history = history_manager.get_history()
        for index, message in enumerate(history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # Only display feedback buttons for assistant messages
                if message["role"] == "assistant":
                    if not message.get("feedback"):
                        col1, col2 = st.columns(2)
                        if col1.button("👍", key=f"thumbsup_{index}"):
                            history_manager.update_feedback(index, "up")
                            st.success("Thanks for your feedback!")
                        if col2.button("👎", key=f"thumbsdown_{index}"):
                            history_manager.update_feedback(index, "down")
                            st.error("Thanks for your feedback!")
                    else:
                        # Display recorded feedback
                        st.info(f"Feedback recorded: {'👍' if message['feedback'] == 'up' else '👎'}")

# Main Application
class ChatApplication:
    def __init__(self):
        self.config = ChatUI.setup_sidebar()
        self.history_manager = ChatHistoryManager(max_length=self.config.max_history_length)
        self.llm = self._initialize_model()
        self.conversation_service = ConversationService(self.llm, self.history_manager) if self.llm else None

    def _initialize_model(self) -> ChatGoogleGenerativeAI:
        print("Here")
        # A simple check to see if the API key is valid (here, starting with "A")
        if not self.config.google_api_key.startswith("A"):
            return None
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=self.config.temperature,
                max_tokens=None,
                timeout=None,
                google_api_key=self.config.google_api_key
            )
        except Exception as e:
            st.error(f"Model Error: {str(e)}")
            return None

    async def run(self):
        # Display the chat history along with any feedback buttons
        ChatUI.display_chat(self.history_manager)

        if not self.llm:
            st.warning("Enter a valid Google API key!")
            return

        # Use Streamlit's built-in chat input
        if prompt := st.chat_input("Ask something"):
            self._process_user_input(prompt)
            await self._generate_and_display_response(prompt)

    def _process_user_input(self, prompt: str):
        with st.chat_message("user"):
            st.write(prompt)
        self.history_manager.add_message("user", prompt)

    async def _generate_and_display_response(self, prompt: str):
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""

            async for chunk in self.conversation_service.generate_stream(prompt):
                full_response += chunk
                response_container.markdown(full_response + "▌")

            response_container.markdown(full_response)
            self.history_manager.add_message("assistant", full_response)
            st.rerun()


# Run Streamlit app
if __name__ == "__main__":
    st.title("AI Chatbot with Memory Checkpointing & Feedback")
    app = ChatApplication()
    asyncio.run(app.run())
