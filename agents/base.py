from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

def create_agent(llm: ChatOpenAI, tools: List[BaseTool], system_prompt: str):
    """Helper function to bind tools to an LLM and add a system prompt."""
    prompt = SystemMessage(content=system_prompt)
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    return llm_with_tools
