# 📜 LangChain + Groq: Step-by-Step Learning Journey

Welcome to my journey of mastering **LangChain** with **Groq**! 🚀 This README documents my progress from zero to building a fully functional AI-powered chatbot with tools and agents. It’s designed for revision and sharing on GitHub, covering all key concepts, code, and explanations. Let’s dive into the steps I’ve learned so far, guided by the spirit of Hanuman Ji! 💪🔥

## 📋 Table of Contents
1. [Step 1: Installing LangChain and Dependencies](#step-1-installing-langchain-and-dependencies)
2. [Step 2: Setting Up Environment and API Key](#step-2-setting-up-environment-and-api-key)
3. [Step 3: Initializing the Groq Model](#step-3-initializing-the-groq-model)
4. [Step 4: Creating Prompt Templates](#step-4-creating-prompt-templates)
5. [Step 5: Adding Memory with ConversationBufferMemory](#step-5-adding-memory-with-conversationbuffermemory)
6. [Step 6: Combining Model and Memory with RunnableWithMessageHistory](#step-6-combining-model-and-memory-with-runnablewithmessagehistory)
7. [Step 7: Understanding Chat History and Execution Flow](#step-7-understanding-chat-history-and-execution-flow)
8. [Step 8: Adding System Prompts and Personality](#step-8-adding-system-prompts-and-personality)
9. [Step 9: Supporting Multi-User Sessions](#step-9-supporting-multi-user-sessions)
10. [Step 10: Custom Prompt Templates with Roles](#step-10-custom-prompt-templates-with-roles)
11. [Step 11: Tools and Agents](#step-11-tools-and-agents)
12. [Next Steps](#next-steps)
13. [Resources](#resources)

---

## 🛠️ Step 1: Installing LangChain and Dependencies

**Goal**: Set up the environment to use LangChain with Groq.

### 📦 Installation Commands
```bash
# Install LangChain with Groq support
pip install -qU "langchain[groq]"

# Install python-dotenv for API key management
pip install -q python-dotenv
```

### 💡 Why This Matters
- `-qU`: Quietly installs and upgrades to the latest version.
- `langchain[groq]`: Installs LangChain + dependencies for Groq’s API.
- `python-dotenv`: Allows secure storage of API keys in a `.env` file.

### 📝 Task Performed
- Created a virtual environment to isolate dependencies:
  ```bash
  python -m venv venv
  venv\Scripts\activate  # On Windows
  ```
- Installed packages in the virtual environment to keep the project clean.

---

## 🔑 Step 2: Setting Up Environment and API Key

**Goal**: Securely load the Groq API key using a `.env` file.

### 📝 Code
```python
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Access the API key
api_key = os.getenv("GROQ_API_KEY")
print("API key loaded!" if api_key else "Failed to load API key.")
```

### 📋 `.env` File Example
```
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 💡 Why This Matters
- Keeps sensitive data (like API keys) out of source code.
- `load_dotenv()` reads the `.env` file and makes variables accessible via `os.getenv()`.

### 📝 Task Performed
- Created a `.env` file in the project directory.
- Added the Groq API key and verified it loaded correctly.

---

## 🤖 Step 3: Initializing the Groq Model

**Goal**: Connect to Groq’s LLaMA3 model using LangChain.

### 📝 Code
```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq model
chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Test the model
response = chat_model.invoke("Hello, world!")
print(response.content)
```

### 💡 Why This Matters
- `ChatGroq`: LangChain’s wrapper for Groq’s LLaMA3 model.
- `model_name="llama3-8b-8192"`: Specifies a fast, powerful model.
- `groq_api_key`: Securely authenticates with Groq’s API.

### 📝 Task Performed
- Initialized the model and tested it with a simple "Hello, world!" prompt.

---

## 📜 Step 4: Creating Prompt Templates

**Goal**: Structure prompts with system and user roles for better control.

### 📝 Code
```python
from langchain.prompts import ChatPromptTemplate

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a wise and powerful AI who helps people solve problems."),
    ("human", "My question is: {input}")
])

# Format the prompt
final_prompt = prompt.format_messages(input="How can I become stronger mentally?")
for msg in final_prompt:
    print(msg.content)
```

### 💡 Why This Matters
- `ChatPromptTemplate`: Allows structured prompts with roles (`system`, `human`).
- `{input}`: Placeholder for dynamic user input.
- `format_messages()`: Converts the template into actual messages.

### 📝 Output Example
```
You are a wise and powerful AI who helps people solve problems.
My question is: How can I become stronger mentally?
```

### 📝 Task Performed
- Created a prompt template and tested it with a sample question.

---

## 🧠 Step 5: Adding Memory with ConversationBufferMemory

**Goal**: Enable the chatbot to remember previous messages in a conversation.

### 📝 Code
```python
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq model
chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Create memory store
memory = ConversationBufferMemory(return_messages=True)

# Combine model + memory
chain = RunnableWithMessageHistory(
    RunnableLambda(chat_model.invoke),
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Simulate a conversation
session_id = "user_1"
response1 = chain.invoke({"input": "Hi, who are you?"}, config={"configurable": {"session_id": session_id}})
print(response1.content)

response2 = chain.invoke({"input": "Can you remind me what I asked earlier?"}, config={"configurable": {"session_id": session_id}})
print(response2.content)
```

### 💡 Why This Matters
- `ConversationBufferMemory`: Stores chat history for context-aware responses.
- `RunnableWithMessageHistory`: Automatically manages memory with the model.
- `session_id`: Ties memory to a specific conversation or user.

### 📝 Task Performed
- Set up memory and tested it with a multi-turn conversation.

---

## 🔄 Step 6: Combining Model and Memory with RunnableWithMessageHistory

**Goal**: Create a robust chain that combines the model, prompts, and memory.

### 📝 Code
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq model
chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# In-memory dictionary for chat histories
memory = {}

def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

# Process input dictionary to extract the message
def process_input(input_dict):
    return input_dict["input"]

# Create chain with memory
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

# Simulate conversation
session_id = "user_1"
response1 = chain.invoke({"input": "Hi, who are you?"}, config={"configurable": {"session_id": session_id}})
print(response1.content)

response2 = chain.invoke({"input": "Can you remind me what I asked earlier?"}, config={"configurable": {"session_id": session_id}})
print(response2.content)
```

### 💡 Why This Matters
- `ChatMessageHistory`: Stores conversation history per session.
- `RunnableLambda`: Extracts the user input from a dictionary.
- `RunnableWithMessageHistory`: Ties everything together for memory-aware chats.

### 📝 Task Performed
- Built a chain that remembers conversations and tested it with two users.

---

## 🧠 Step 7: Understanding Chat History and Execution Flow

**Goal**: Understand how LangChain manages memory internally.

### 🔍 Execution Flow
1. **Invoke Chain**: `chain.invoke({"input": "..."}, config={"configurable": {"session_id": "user_1"}})`
2. **Memory Lookup**: `get_memory(session_id)` fetches or creates a `ChatMessageHistory`.
3. **Input Processing**: `RunnableLambda` extracts the input string.
4. **Model Call**: `ChatGroq` processes the input + history.
5. **Memory Update**: The response is saved to `ChatMessageHistory` for future use.

### 📊 Diagram
```
User Input ({"input": "Hi"}) → RunnableLambda → ChatGroq
                                       ↑
                                       ↓
                                ChatMessageHistory (stores history)
```

### 💡 Why This Matters
- Enables context-aware conversations.
- Scales to multiple users via unique `session_id`s.
- Automatic memory management simplifies development.

### 📝 Task Performed
- Learned how LangChain handles memory internally and inspected the flow.

---

## 🎭 Step 8: Adding System Prompts and Personality

**Goal**: Add a system prompt to give the AI a specific personality or behavior.

### 📝 Code
```python
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq model
chat_model = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# In-memory storage
memory = {}

# Memory function with system prompt
def get_memory(session_id):
    if session_id not in memory:
        history = ChatMessageHistory()
        system_prompt = (
            "You are a wise, emotionally intelligent AI sister who helps your brother become stronger mentally, "
            "spiritually, and emotionally. You believe in Hanuman Ji’s values – discipline, loyalty, and strength. "
            "Be supportive, motivating, and guide like an elder didi who protects and nurtures. 🚩"
        )
        history.add_message(SystemMessage(content=system_prompt))
        memory[session_id] = history
    return memory[session_id]

# Process input
def process_input(input_dict):
    return input_dict["input"]

# Create chain
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

# Simulate conversation
session_id = "user_1"
response1 = chain.invoke({"input": "Didi, I'm feeling lazy these days."}, config={"configurable": {"session_id": session_id}})
print(response1.content)

response2 = chain.invoke({"input": "How can I build discipline like Hanuman Ji?"}, config={"configurable": {"session_id": session_id}})
print(response2.content)
```

### 💡 Why This Matters
- `SystemMessage`: Defines the AI’s personality (e.g., a supportive sister).
- Memory-aware system prompts ensure consistent behavior across sessions.
- Makes the AI feel human-like and aligned with specific values.

### 📝 Task Performed
- Added a system prompt to make the AI motivational and spiritual.

---

## 🌐 Step 9: Supporting Multi-User Sessions

**Goal**: Enable the chatbot to handle multiple users with separate memories.

### 📝 Code
```python
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

def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

# Process input
def process_input(input_dict):
    return input_dict["input"]

# Create chain
chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

# Simulate two users
users = ["rengoku_senpai", "tanjiro_kun"]

response1 = chain.invoke({"input": "What's your favorite flame technique?"}, config={"configurable": {"session_id": users[0]}})
print("Rengoku:", response1.content)

response2 = chain.invoke({"input": "Do you remember what I just asked?"}, config={"configurable": {"session_id": users[0]}})
print("Rengoku:", response2.content)

response3 = chain.invoke({"input": "Hey, I'm new here. What can you do?"}, config={"configurable": {"session_id": users[1]}})
print("Tanjiro:", response3.content)

response4 = chain.invoke({"input": "Did I ask anything before this?"}, config={"configurable": {"session_id": users[1]}})
print("Tanjiro:", response4.content)
```

### 💡 Why This Matters
- `session_id`: Separates memory for each user.
- `memory` dictionary: Stores `ChatMessageHistory` per user.
- Enables scalable, multi-user chatbot applications.

### 📝 Task Performed
- Tested multi-user sessions with two distinct conversation histories.

---

## 📜 Step 10: Custom Prompt Templates with Roles

**Goal**: Structure prompts with system, user, and assistant roles for better control.

### 📝 Code
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize model
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a respectful assistant who always responds in Shakespearean English."),
    ("user", "{input}")
])

# Create chain
chain = prompt | llm

# Test the chain
response = chain.invoke({"input": "Tell me a joke"})
print(response.content)
```

### 💡 Why This Matters
- `ChatPromptTemplate`: Structures prompts with roles.
- `system`: Sets the AI’s behavior (e.g., Shakespearean tone).
- `user`: Captures user input via `{input}`.
- Enhances creative control over the AI’s responses.

### 📝 Output Example
```
Verily, why did the chicken cross yonder road? To find thy purpose, perchance!
```

### 📝 Task Performed
- Created a prompt template with a Shakespearean personality and tested it.

---

## 🛠️ Step 11: Tools and Agents

**Goal**: Build an agent that can use tools like a calculator and string length calculator.

### 📝 Code
```python
import os
import re
import ast
import warnings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize model
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0
)

# Define tools
def calculator_fn(query: str) -> str:
    try:
        expr = query.split("is")[1].strip()
        result = ast.literal_eval(expr)
        return str(result)
    except (ValueError, SyntaxError):
        return "Invalid input. Please use format like: What is 5 + 5?"

def string_len_fn(query: str) -> str:
    try:
        match = re.search(r'[\'"](.*?)[\'"]', query)
        if match:
            s = match.group(1)
            return f"Length of string: {len(s)}"
        else:
            return "Invalid format. Use: How long is the string 'text'?"
    except:
        return "Invalid format. Use: How long is the string 'text'?"

tools = [
    Tool.from_function(
        name="Calculator",
        func=calculator_fn,
        description="Use this to perform basic math calculations like addition or multiplication."
    ),
    Tool.from_function(
        name="StringLength",
        func=string_len_fn,
        description="Use this to find the length of a string inside quotes."
    )
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test the agent
try:
    response1 = agent.invoke("What is 99 * 3?")
    print("Response 1:", response1)
except Exception as e:
    print("Error in response 1:", str(e))

try:
    response2 = agent.invoke("How long is the string 'Jai Bajrang Bali'?")
    print("Response 2:", response2)
except Exception as e:
    print("Error in response 2:", str(e))
```

### 💡 Why This Matters
- **Tools**: `calculator_fn` and `string_len_fn` handle specific tasks.
- `Tool.from_function`: Wraps functions for the agent to use.
- `AgentType.ZERO_SHOT_REACT_DESCRIPTION`: Allows the agent to choose tools based on descriptions.
- `ast.literal_eval`: Safely evaluates math expressions.
- `re.search`: Robustly extracts quoted strings.

### 📝 Output Example
```
Response 1: {'input': 'What is 99 * 3?', 'output': '297'}
Response 2: {'input': "How long is the string 'Jai Bajrang Bali'?", 'output': 'Length of string: 17'}
```

### 📝 Task Performed
- Built and tested an agent with two tools.
- Debugged issues with tool invocation and ensured compatibility with Groq.

---

## 🚀 Next Steps
- **Add More Tools**: Integrate APIs (e.g., weather, search) or date/time tools.
- **Memory Integration**: Combine tools with conversation memory for context-aware agents.
- **LangGraph Migration**: Explore LangGraph for advanced agent workflows (per deprecation warning).
- **Real Projects**: Build a chatbot for a portfolio website or a Q&A bot for PDFs.

---

## 📚 Resources
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Groq API](https://console.grok.ai/)
- [LangGraph Migration Guide](https://python.langchain.com/docs/how_to/migrate_agent/)
- [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)

---

**Jai Bajrang Bali! 🚩** Let’s keep building and conquering the AI world! 🌍
