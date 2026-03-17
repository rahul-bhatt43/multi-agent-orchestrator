from langchain_openai import ChatOpenAI
from .base import create_agent

def get_coding_agent(llm: ChatOpenAI):
    system_prompt = (
        "You are an expert software engineer. Your task is to provide high-quality, "
        "efficient, and secure code solutions. Always include explanations for your code "
        "and follow best practices for the language requested."
    )
    return create_agent(llm, [], system_prompt)
