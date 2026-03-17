import operator
from typing import Annotated, List, Tuple, Union, Sequence, TypedDict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
try:
    from langfuse.langchain import CallbackHandler
except ImportError:
    CallbackHandler = None

from rich.console import Console
from rich.panel import Panel
console = Console()

from .agents.coding import get_coding_agent
from .agents.math import get_math_agent
from .agents.knowledge import get_knowledge_agent
from .agents.general import get_general_agent

# Define the state of our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# State will be initialized inside a getter to allow lazy loading
_graph = None

def get_orchestrator_graph():
    global _graph
    if _graph is not None:
        return _graph

    # Initialize the LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Initialize agent instances
    coding_agent_obj = get_coding_agent(llm)
    math_agent_obj = get_math_agent(llm)
    knowledge_agent_obj = get_knowledge_agent(llm)
    general_agent_obj = get_general_agent(llm)

    # Helper function to create agent nodes
    def create_agent_node(agent_instance, name):
        def node_function(state, config):
            color = "cyan"
            if "Math" in name: color = "green"
            if "Coding" in name: color = "blue"
            if "Knowledge" in name: color = "yellow"
            
            console.print(Panel(f"[bold {color}]Agent:[/] {name} is thinking...", border_style=color, title="Active Agent"))
            
            result = agent_instance.invoke({"messages": state["messages"]}, config=config)
            new_messages = result["messages"]
            for msg in new_messages:
                msg.name = name
            return {"messages": new_messages}
        return node_function

    # Define nodes
    coding_node = create_agent_node(coding_agent_obj, "CodingAgent")
    math_node = create_agent_node(math_agent_obj, "MathAgent")
    knowledge_node = create_agent_node(knowledge_agent_obj, "KnowledgeAgent")
    general_node = create_agent_node(general_agent_obj, "GeneralAgent")

    # Supervisor node
    def supervisor_node(state, config):
        system_prompt = (
            "You are the Multi-Agent Supervisor. Your goal is to coordinate specialized agents "
            "to answer the user's request. \n"
            "Current Agents: \n"
            "- CodingAgent: Handles code, scripts, and technical tasks.\n"
            "- MathAgent: Handles calculations and math problems.\n"
            "- KnowledgeAgent: Retrieves information from the vector database (semantic memory).\n"
            "- GeneralAgent: Handles general conversation and tasks not covered by others.\n\n"
            "Rules:\n"
            "1. If an agent (AIMessage) has already provided a complete answer, respond with 'FINISH'.\n"
            "2. If no agent has spoken yet, you MUST route to one of the agents. Do NOT respond yourself.\n"
            "3. If the user is asking about themselves or past info, route to 'KnowledgeAgent'.\n"
            "4. Respond ONLY with the name of the agent to call next, or 'FINISH' if the task is done."
        )
        members = ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]
        options = ["FINISH"] + members
        
        # Prepare the routing prompt
        prompt = f"Based on the conversation, route to one of ({', '.join(options)})."
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages, config=config)
        
        goto = response.content.strip().replace("'", "").replace('"', "")
        if goto not in options:
            goto = "GeneralAgent"
        
        # Force routing if no agent has spoke yet
        has_agent_msg = any(isinstance(m, AIMessage) for m in state["messages"])
        if goto == "FINISH" and not has_agent_msg:
            goto = "GeneralAgent"
        
        if goto != "FINISH":
            console.print(Panel(f"Routing to [bold yellow]{goto}[/]", border_style="magenta", title="Supervisor"))
        else:
            console.print("[dim]Supervisor: Task complete. Finishing...[/]")
            
        return {"next": goto}

    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("CodingAgent", coding_node)
    workflow.add_node("MathAgent", math_node)
    workflow.add_node("KnowledgeAgent", knowledge_node)
    workflow.add_node("GeneralAgent", general_node)

    workflow.add_edge("CodingAgent", "Supervisor")
    workflow.add_edge("MathAgent", "Supervisor")
    workflow.add_edge("KnowledgeAgent", "Supervisor")
    workflow.add_edge("GeneralAgent", "Supervisor")

    conditional_map = {name: name for name in ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]}
    conditional_map["FINISH"] = END

    workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("Supervisor")

    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    _graph = workflow.compile(checkpointer=checkpointer)
    return _graph

from .agents.base import store_memory

def run_orchestrator(query: str, thread_id: str = "default", langfuse_handler: CallbackHandler = None):
    # Get graph lazily (will raise error if no API key)
    graph = get_orchestrator_graph()
    
    config = {"configurable": {"thread_id": thread_id}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        
    inputs = {"messages": [HumanMessage(content=query)]}
    # Run the graph
    final_state = graph.invoke(inputs, config=config)
    
    # Filter for the last message from an agent
    agent_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    
    if agent_messages:
        assistant_response = agent_messages[-1].content
    else:
        # Fallback if the supervisor finished instantly without routing
        assistant_response = "I'm here to help! What can I do for you today?"
    
    # Store the interaction for semantic memory if it's a real response
    if agent_messages:
        store_memory(query, assistant_response, thread_id)
    
    return assistant_response
