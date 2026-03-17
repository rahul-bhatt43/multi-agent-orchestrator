import operator
from typing import Annotated, List, Tuple, Union, Sequence, TypedDict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
try:
    from langfuse.callback import CallbackHandler
except ImportError:
    CallbackHandler = None

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
        def node_function(state):
            result = agent_instance.invoke({"messages": state["messages"]})
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
    def supervisor_node(state):
        system_prompt = (
            "You are the Source Control Agent (Supervisor). Your job is to route the user's "
            "request to the most appropriate agent: CodingAgent, MathAgent, KnowledgeAgent, or GeneralAgent. "
            "If you have enough information to answer the user, or if all agents have completed their tasks, "
            "respond with FINISH."
        )
        members = ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]
        options = ["FINISH"] + members
        prompt = f"Route the conversation to one of: {', '.join(options)}. Respond ONLY with the name of the agent or FINISH."
        
        messages = [HumanMessage(content=system_prompt + "\n\n" + prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        goto = response.content.strip()
        if goto not in options:
            goto = "GeneralAgent"
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
    final_state = graph.invoke(inputs, config=config)
    
    assistant_response = final_state["messages"][-1].content
    
    # Store the interaction for semantic memory
    store_memory(query, assistant_response, thread_id)
    
    return assistant_response
