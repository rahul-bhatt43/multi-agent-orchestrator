import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from .base import create_agent

@tool
def query_knowledge_base(query: str):
    """Query the Pinecone vector database for information."""
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

def get_knowledge_agent(llm: ChatOpenAI):
    tools = [query_knowledge_base]
    system_prompt = (
        "You are a knowledge retrieval expert. Your task is to provide accurate information "
        "by searching the knowledge base. Always ground your answers in the retrieved content."
    )
    return create_agent(llm, tools, system_prompt)
