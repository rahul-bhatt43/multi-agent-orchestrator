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
    if os.getenv("SEMANTIC_MEMORY_ENABLED") != "true":
        return
        
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        api_key = os.getenv("PINECONE_API_KEY")
        if not index_name or not api_key:
            return
            
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)
        
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
    if os.getenv("SEMANTIC_MEMORY_ENABLED") != "true":
        return ""
        
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        api_key = os.getenv("PINECONE_API_KEY")
        if not index_name or not api_key:
            return ""
            
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)
        
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""
            
        memory_text = "\n---\n".join([doc.page_content for doc in docs])
        return f"\nRelevant past interactions:\n{memory_text}\n"
    except Exception as e:
        # Silently fail for retrieval if memory is not working
        return ""

def create_agent(llm: ChatOpenAI, tools: List[Union[BaseTool, callable]], system_prompt: str):
    """
    Returns an Agent instance that handles system prompt and tool execution.
    """
    class Agent:
        def __init__(self, llm: ChatOpenAI, tools: List[Union[BaseTool, callable]], system_prompt: str):
            self.llm = llm
            self.tools = tools
            self.system_prompt = system_prompt

        def agent_node(self, inputs: dict, config: dict = None):
            """Standard agent node for LangGraph."""
            messages = inputs.get("messages", [])
            user_query = messages[-1].content if messages else ""
            
            # Retrieve long-term semantic context
            semantic_context = retrieve_memories(user_query)
            
            # Prepare messages with system prompt + semantic context
            full_system_prompt = self.system_prompt
            if semantic_context:
                full_system_prompt += f"\n\n{semantic_context}\nUse the above context if relevant."
            
            # Ensure the system message is prepended
            combined_messages = [SystemMessage(content=full_system_prompt)] + list(messages)
            
            try:
                if self.tools:
                    llm_with_tools = self.llm.bind_tools(self.tools)
                else:
                    llm_with_tools = self.llm
                
                while True:
                    # Pass config for Langfuse tracing
                    response = llm_with_tools.invoke(combined_messages, config=config)
                    
                    if not response.tool_calls:
                        # Store in semantic memory if enabled
                        thread_id = "default"
                        if config and "configurable" in config:
                            thread_id = config["configurable"].get("thread_id", "default")
                        store_memory(user_query, response.content, thread_id)
                        
                        return {"messages": [response]}
                    
                    combined_messages.append(response)
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool = next((t for t in self.tools if (getattr(t, "name", None) == tool_name or getattr(t, "__name__", None) == tool_name)), None)
                        
                        if tool:
                            try:
                                tool_output = tool.invoke(tool_args) if hasattr(tool, "invoke") else tool(**tool_args)
                                combined_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                            except Exception as e:
                                combined_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"]))
                        else:
                            combined_messages.append(ToolMessage(content=f"Tool {tool_name} not found.", tool_call_id=tool_call["id"]))
            except Exception as e:
                print(f"DEBUG: Error in agent: {e}")
                return {"messages": [AIMessage(content=f"An error occurred: {e}")]}

        def invoke(self, inputs: dict, config: dict = None):
            return self.agent_node(inputs, config=config)

    return Agent(llm, tools, system_prompt)