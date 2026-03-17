from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from .base import create_agent

def get_math_agent(llm: ChatOpenAI):
    tools = load_tools(["llm-math"], llm=llm)
    system_prompt = (
        "You are a mathematical assistant. Your goal is to solve complex math problems "
        "step-by-step. Use the calculator tool for precise calculations."
    )
    
    return create_agent(llm, tools, system_prompt)