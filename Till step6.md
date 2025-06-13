# LangChain + Groq Step-by-Step Guide (Till Step 6)

This guide explains everything we’ve covered so far in your journey to becoming a **LangChain + Groq pro developer**. Use this as a revision or reference guide when needed.

---

## ✅ Step 1: Installing and Setting Up Environment

### 1.1 Install LangChain with Groq Support

```bash
pip install -U "langchain[groq]"
```

### 1.2 Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 1.3 Install Dotenv for API Key Management

```bash
pip install python-dotenv
```

### 1.4 Create a `.env` File

Create a `.env` file in your project directory and add your Groq API key:

```
GROQ_API_KEY=your_actual_api_key_here
```

---

## ✅ Step 2: Initializing Groq API Model

```python
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
```

* **model**: We’re using LLaMA3 from Groq.
* **api\_key**: Loaded securely using `dotenv`.

---

## ✅ Step 3: Creating Prompt Templates

We used static string input in early steps. Prompt templates will be revisited later (function-calling, system+human prompts).

---

## ✅ Step 4: Using Memory — `ConversationBufferMemory`

This was updated in recent LangChain versions. Instead of accessing `.messages`, memory is now auto-managed via `RunnableWithMessageHistory`.

---

## ✅ Step 5: Combining Model + Prompt Using `RunnableWithMessageHistory`

We combined all pieces into one chain with memory support.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

memory = {}

def get_memory(session_id):
    if session_id not in memory:
        memory[session_id] = ChatMessageHistory()
    return memory[session_id]

def process_input(input_dict):
    return input_dict["input"]

chain = RunnableWithMessageHistory(
    RunnableLambda(process_input) | chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)

session_id = "user_1"

response1 = chain.invoke(
    {"input": "Hi, who are you?"},
    config={"configurable": {"session_id": session_id}}
)
print(response1.content)

response2 = chain.invoke(
    {"input": "Can you remind me what I asked earlier?"},
    config={"configurable": {"session_id": session_id}}
)
print(response2.content)
```

---

## ✅ Step 6: Memory Structure Explained

| 🔧 Component                 | 💡 Purpose                                            |
| ---------------------------- | ----------------------------------------------------- |
| `ChatGroq`                   | Connects to LLaMA3 model from Groq API                |
| `RunnableLambda`             | Converts dict to plain text string                    |
| `ChatMessageHistory`         | Stores previous messages in a session                 |
| `RunnableWithMessageHistory` | Combines model + memory for full conversational state |
| `invoke()`                   | Sends input and fetches memory-aware output           |

This allows the model to **remember past messages** within a session.

---

✅ You are now ready for: **Step 7: Understanding how chat history is managed and how memory flows inside the chain.**
