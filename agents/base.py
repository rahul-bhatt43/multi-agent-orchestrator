from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

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
        combined_messages = [SystemMessage(content=system_prompt)] + messages
        
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
            
            # Loop continues to invoke LLM with tool results
            
    return type("AgentRunnable", (), {"invoke": agent_node})()