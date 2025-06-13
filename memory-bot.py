# Import necessary modules
from langchain_community.chat_message_histories import ChatMessageHistory  # For storing chat history
from langchain_core.runnables.history import RunnableWithMessageHistory  # To wrap chains with memory
from langchain_groq import ChatGroq  # LLM from Groq
from langchain_core.runnables import RunnableLambda  # To apply custom logic in chains
from dotenv import load_dotenv  # For loading environment variables
import os

# Load environment variables from .env file (like GROQ_API_KEY)
load_dotenv()

# Initialize the LLM with Groq using LLaMA3 model
chat_model = ChatGroq(
    model="llama3-8b-8192",  # Model name
    api_key=os.getenv("GROQ_API_KEY")  # API key loaded from .env
)

# Create a dictionary to store memory for different user sessions
memory = {}  # Format: {session_id: ChatMessageHistory()}

# This function returns existing memory or creates new if not present
def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()  # Store a fresh chat history
    return memory[session_id]

# Extract user message from input dictionary
def process_input(input_dict):
    return input_dict["input"]  # Returns only the string input

# Combine the input processor and model into a single chain with memory
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,  # Pipe input to model
    get_memory,  # Memory function to fetch history based on session_id
    input_messages_key="input",  # Key from input dict
    history_messages_key="chat_history",  # Internal memory tracking
    output_messages_key="output"  # Optional: name of output key
)

# Define session ID (used to separate chat histories per user)
session_id = "user_1"

# First input to the chain
response1 = chain.invoke(
    {"input": "Hi, who are you?"},  # The message user sends
    config={"configurable": {"session_id": session_id}}  # Pass session info
)
print(response1.content.strip())  # Show model's reply

# Second input to the same session (it should remember the previous message)
response2 = chain.invoke(
    {"input": "Can you remind me what I asked earlier?"},  # Follow-up question
    config={"configurable": {"session_id": session_id}}  # Same session
)
print(response2.content.strip())  # Show response with memory
