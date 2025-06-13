# Import necessary classes to handle messages (system, human, etc.)
from langchain_core.messages import SystemMessage, HumanMessage

# LangChain's utility to initialize a chat model
from langchain.chat_models import init_chat_model

# For loading your GROQ_API_KEY from .env file
from dotenv import load_dotenv
import os

# Load environment variables (like your Groq API key) from .env
load_dotenv()

# ---------------- STEP 1: Initialize your model ----------------
model = init_chat_model(
    model="llama3-8b-8192",       # The LLaMA3 model via Groq
    model_provider="groq",        # We're using Groq as backend
    temperature=0.7,              # Creativity level (0 = factual, 1 = creative)
    max_tokens=300                # Limit the number of tokens in the output
)

# ---------------- STEP 2: Prepare your chat history ----------------
chat_history = [
    SystemMessage(
        content="You are a calm, focused AI who explains everything with depth and clarity."
    ),
    HumanMessage(
        content="Explain what is Karma according to Hinduism."
    )
]
# Note:
# - SystemMessage = Defines the assistant's personality.
# - HumanMessage = User's input (like your question).
# - You can add previous messages to simulate a conversation context.

# ---------------- STEP 3: Streaming the Response ----------------

# This loop streams the AI’s reply one chunk at a time — just like ChatGPT typing
for chunk in model.stream(chat_history):
    # Each 'chunk' is a partial piece of the model's response (like a word or phrase)
    print(chunk.content, end="", flush=True)

# Breakdown of print line:
# - chunk.content : This is the actual text response piece from the model.
# - end="" : Prevents newline after every chunk (so text comes in one line).
# - flush=True : Forces the output to show immediately without waiting (good for real-time feel).
