import os
import re
import ast
import warnings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Suppress deprecation warning (optional)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Step 1: Define the LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0
    # groq_api_key="your-api-key"  # Uncomment and add your key if not using environment variable
)

# Step 2: Create Tool Functions
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

# Step 3: Register Tools
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

# Step 4: Create the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 5: Try Out the Agent
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