# Filename: FileName.py

from langchain.agents import Tool, AgentExecutor
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from sympy import sympify
import os

# Load environment variables
load_dotenv()

# --------------------------
# 🔧 Step 1: Define Tools
# --------------------------

def calculator_tool(query: str) -> str:
    try:
        result = sympify(query).evalf()
        return f"The result is {result}"
    except:
        return "Invalid expression"

def string_length_tool(text: str) -> str:
    return f"The length of the string is {len(text)} characters."

tools = [
    Tool(
        name="CalculatorTool",
        func=calculator_tool,
        description="Evaluates basic math expressions. Input must be a valid math expression like '12 * 8'."
    ),
    Tool(
        name="StringLengthTool",
        func=string_length_tool,
        description="Counts the length of a string. Input should be a phrase like 'Hanuman Bhakt'."
    )
]

# --------------------------
# 🧠 Step 2: Initialize LLM
# --------------------------

llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# --------------------------
# 🤖 Step 3: Create Agent with Custom Prompt
# --------------------------

prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Use the provided tools ({tool_names}) to answer the query concisely. Call each tool only once unless explicitly needed. Always respond with a JSON object containing "action" and "action_input" fields. For tool calls, set "action" to the tool name and "action_input" to the tool's input. For the final answer, set "action" to "Final Answer" and "action_input" to the answer string.

Example:
- For tool call: ```{"action": "CalculatorTool", "action_input": "12 * 8"}```
- For final answer: ```{"action": "Final Answer", "action_input": "The result is 96"}```

Query: {input}
Tools: {tools}
Agent Scratchpad: {agent_scratchpad}"""
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=2,
    handle_parsing_errors=True  # Allow retry on parsing errors
)

# --------------------------
# 🧪 Step 4: Run Tests
# --------------------------

# Math test
response1 = agent_executor.invoke({"input": "What is 99 * 3?"})
print("Answer 1:", response1['output'])

# String test
response2 = agent_executor.invoke({"input": "How long is the string 'Hanuman Bhakt'?"})
print("Answer 2:", response2['output'])