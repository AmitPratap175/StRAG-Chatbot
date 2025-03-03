import asyncio
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Generator, TypedDict

# Configuration
class AppConfig(BaseModel):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    temperature: float = 0.5
    max_history_length: int = 10

# State definition
class ChatState(TypedDict):
    messages: List[Dict[str, str]]

# Chat history manager
class ChatHistoryManager:
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        if "history" not in st.session_state:
            st.session_state.history = []

    def add_message(self, role: str, content: str):
        st.session_state.history.append({"role": role, "content": content})
        if len(st.session_state.history) > self.max_length:
            st.session_state.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        return st.session_state.history

    def clear_history(self):
        st.session_state.history = []

# LangGraph conversation service
class ConversationService:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
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
            
            # Add new user message to the state
            new_state = {"messages": [{"role": "user", "content": input_text}]}
            
            # Stream the response
            async for event in self.app.astream(
                new_state,
                config=config,
                stream_mode="values"
            ):
                if event and "assistant" in event:
                    response = event["assistant"]["messages"][-1]["content"]
                    yield response
                    await asyncio.sleep(0.02)
                    
        except Exception as e:
            yield f"Error: {str(e)}"

# Streamlit UI
class ChatUI:
    @staticmethod
    def setup_sidebar() -> AppConfig:
        st.sidebar.title("Settings")
        return AppConfig(
            openai_api_key=st.sidebar.text_input("OpenAI API Key", type="password"),
            temperature=st.sidebar.slider("Temperature", 0.0, 1.0, 0.5),
        )

    @staticmethod
    def display_chat(history_manager: ChatHistoryManager):
        for message in history_manager.get_history():
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Main Application
class ChatApplication:
    def __init__(self):
        self.config = ChatUI.setup_sidebar()
        self.history_manager = ChatHistoryManager()
        self.llm = self._initialize_model()
        self.conversation_service = ConversationService(self.llm) if self.llm else None

    def _initialize_model(self) -> ChatOpenAI:
        if not self.config.openai_api_key.startswith("sk-"):
            return None
        try:
            return ChatOpenAI(
                temperature=self.config.temperature,
                openai_api_key=self.config.openai_api_key,
                model="gpt-3.5-turbo"
            )
        except Exception as e:
            st.error(f"Model Error: {str(e)}")
            return None

    async def run(self):
        ChatUI.display_chat(self.history_manager)

        if not self.llm:
            st.warning("Enter a valid OpenAI API key!")
            return

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
    st.title("AI Chatbot with Memory Checkpointing")
    app = ChatApplication()
    asyncio.run(app.run())