import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Step 1: Load API key
load_dotenv()

# Step 2: Initialize Groq model (Llama3)
model = init_chat_model(
    model="llama3-8b-8192",
    model_provider="groq",
    temperature=0.7,
    max_tokens=300
)
# Extra step:  Define the system behavior
system_prompt = SystemMessage(
    content="You are Bajrang, an extremely powerful, wise, and emotionally intelligent guide who explains everything to the user like a calm and devoted warrior of Dharma."
)
# Step 3: Initialize chat history
chat_history = []
# Extra step: Add system message at the beginning of chat history
chat_history = [system_prompt]

print("⚡ Ask me anything about Ramayana, Hanuman, or life! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chat ended. Jai Bajrangbali! 🚩")
        break

    # Step 4: Add your message to history
    chat_history.append(HumanMessage(content=user_input))

    # Step 5: Get model response
    response = model.invoke(chat_history)

    # Step 6: Print AI reply and add it to history
    print("AI:", response.content.strip())
    chat_history.append(AIMessage(content=response.content.strip()))
