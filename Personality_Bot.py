from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

memory = {}

# Create memory for a session
def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

# Convert the dict input to string (just the "input" key)
def process_input(input_dict):
    return input_dict["input"]

# Define a system message dynamically
def system_message_func(config):
    return SystemMessage(
        content=(
            "You are a wise, emotionally intelligent, and powerful AI sister "
            "who helps your brother become stronger mentally, spiritually, and emotionally. "
            "You believe in the values of Hanuman ji – discipline, loyalty, and strength. "
            "Be supportive, motivating, and guide him like an elder didi who protects and nurtures. "
            "Encourage him to focus, be fearless, and always walk the path of righteousness 🚩."
        )
    )

# Wrap the full runnable
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
    history_factory_configurable={"system_message": system_message_func}
)

# Simulate conversation
session_id = "user_1"

response1 = chain.invoke(
    {"input": "Didi, I'm feeling lazy these days."},
    config={"configurable": {"session_id": session_id}}
)
print(response1.content)

response2 = chain.invoke(
    {"input": "How can I build discipline like Hanuman ji?"},
    config={"configurable": {"session_id": session_id}}
)
print(response2.content)
