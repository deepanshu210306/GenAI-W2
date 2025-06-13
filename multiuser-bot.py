from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize model
chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# In-memory storage for all users
memory = {}

# Return existing or new message history per user
def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

# Extract input text
def process_input(input_dict):
    return input_dict["input"]

# Create the chain with memory awareness
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

# Simulating Two Users Talking to the Same Bot

users = ["rengoku_senpai", "tanjiro_kun"]

# Rengoku's conversation
response1 = chain.invoke(
    {"input": "What's your favorite flame technique?"},
    config={"configurable": {"session_id": users[0]}}
)
print("🔥 Rengoku:", response1.content)

response2 = chain.invoke(
    {"input": "Do you remember what I just asked?"},
    config={"configurable": {"session_id": users[0]}}
)
print("🔥 Rengoku:", response2.content)

# Tanjiro starts his chat (new memory!)
response3 = chain.invoke(
    {"input": "Hey, I'm new here. What can you do?"},
    config={"configurable": {"session_id": users[1]}}
)
print("🌊 Tanjiro:", response3.content)

response4 = chain.invoke(
    {"input": "Did I ask anything before this?"},
    config={"configurable": {"session_id": users[1]}}
)
print("🌊 Tanjiro:", response4.content)
