# 📦 Load libraries
import os
from dotenv import load_dotenv  # used to load .env file
from langchain.chat_models import init_chat_model  # to initialize the chat model from LangChain

# 📩 Messages for the chatbot
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ✅ Step 1: Load the .env file (this loads the GROQ_API_KEY into memory)
load_dotenv()
# This line extracts the API key from your `.env` file
# It makes the key available as: os.environ["GROQ_API_KEY"]

# ✅ Step 2: Initialize the LLM with Groq using LangChain's official syntax
model = init_chat_model(
    model="llama3-8b-8192",        # which model to use from Groq
    model_provider="groq",         # name of the provider
    temperature=0.7,               # controls creativity; higher = more creative
    max_tokens=300                 # max tokens the model can return
)

# ✅ Step 3: Create chat history (like a conversation log)
chat_history = []  # a list to store back-and-forth messages

# Add a Human message (question)
chat_history.append(HumanMessage(content="Who is Hanuman?"))

# Add AI's earlier reply (simulate ongoing convo)
chat_history.append(AIMessage(content="Hanuman is a powerful devotee of Lord Rama."))

# Add next Human question
chat_history.append(HumanMessage(content="What was his role in the Ramayana?"))

# ✅ Step 4: Send chat history to the model (like you are chatting with memory)
response = model.invoke(chat_history)

# ✅ Step 5: Print the model's reply
print(response.content.strip())  # .strip() removes extra spaces or new lines
