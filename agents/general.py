from langchain_openai import ChatOpenAI
from .base import create_agent

def get_general_agent(llm: ChatOpenAI):
    system_prompt = (
        "You are a helpful, general-purpose AI assistant. Your goal is to answer "
        "any questions concisely and accurately. Use your broad knowledge base to help the user."
    )
    return create_agent(llm, [], system_prompt)
