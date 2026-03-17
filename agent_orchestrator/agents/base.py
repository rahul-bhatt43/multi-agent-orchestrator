import os
from datetime import datetime
from typing import List, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

def store_memory(query: str, response: str, thread_id: str):
    """Stores a query-response pair into Pinecone for long-term memory."""
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            return
            
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
        doc = Document(
            page_content=f"User: {query}\nAssistant: {response}",
            metadata={
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat(),
                "type": "chat_history"
            }
        )
        vectorstore.add_documents([doc])
    except Exception as e:
        print(f"DEBUG: Error storing memory: {e}")

def retrieve_memories(query: str, k: int = 5):
    """Retrieves relevant past interactions from Pinecone."""
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            return ""
            
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""
            
        memory_text = "\n---\n".join([doc.page_content for doc in docs])
        return f"\nRelevant past interactions:\n{memory_text}\n"
    except Exception as e:
        print(f"DEBUG: Error retrieving memory: {e}")
        return ""

def create_agent(llm: ChatOpenAI, tools: List[Union[BaseTool, callable]], system_prompt: str):
    """
    Returns a 'Runnable' that handles system prompt and tool execution.
    """
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    def agent_node(self, inputs: dict):
        messages = inputs.get("messages", [])
        user_query = messages[-1].content if messages else ""
        
        # Retrieve long-term semantic context
        semantic_context = retrieve_memories(user_query)
        
        full_system_prompt = system_prompt
        if semantic_context:
            full_system_prompt += f"\n\n{semantic_context}\nUse the above context if relevant to the current conversation."
            
        combined_messages = [SystemMessage(content=full_system_prompt)] + messages
        
        while True:
            response = llm_with_tools.invoke(combined_messages)
            
            # If no tool calls, return the response
            if not response.tool_calls:
                return {"messages": [response]}
            
            # Execute tool calls
            combined_messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find the tool
                tool = next((t for t in tools if (getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name)), None)
                if tool:
                    try:
                        tool_output = tool.invoke(tool_args) if hasattr(tool, "invoke") else tool(**tool_args)
                        combined_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                    except Exception as e:
                        combined_messages.append(ToolMessage(content=f"Error executing tool: {str(e)}", tool_call_id=tool_call["id"]))
                else:
                    combined_messages.append(ToolMessage(content=f"Tool {tool_name} not found.", tool_call_id=tool_call["id"]))
            
    return type("AgentRunnable", (), {"invoke": agent_node})()