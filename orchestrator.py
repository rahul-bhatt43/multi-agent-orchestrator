import operator
from typing import Annotated, List, Tuple, Union, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langfuse.callback import CallbackHandler

from agents.coding import get_coding_agent
from agents.math import get_math_agent
from agents.knowledge import get_knowledge_agent
from agents.general import get_general_agent

# Define the state of our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create agents
coding_agent = get_coding_agent(llm)
math_agent = get_math_agent(llm)
knowledge_agent = get_knowledge_agent(llm)
general_agent = get_general_agent(llm)

# Helper function to create agent nodes
def create_node(agent, name):
    def node(state):
        result = agent.invoke(state["messages"])
        return {
            "messages": [AIMessage(content=result.content, name=name)],
        }
    return node

# Define nodes
coding_node = create_node(coding_agent, "CodingAgent")
math_node = create_node(math_agent, "MathAgent")
knowledge_node = create_node(knowledge_agent, "KnowledgeAgent")
general_node = create_node(general_agent, "GeneralAgent")

# Supervisor (Source Control Agent)
def supervisor_node(state):
    system_prompt = (
        "You are the Source Control Agent (Supervisor). Your job is to route the user's "
        "request to the most appropriate agent: CodingAgent, MathAgent, KnowledgeAgent, or GeneralAgent. "
        "If you have enough information to answer the user, or if all agents have completed their tasks, "
        "respond with FINISH."
    )
    
    # We use function calling to decide where to go
    members = ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]
    options = ["FINISH"] + members
    
    # In a real scenario, we'd bind a choice tool. 
    # For simplicity, we'll use a specific format or an LLM call.
    prompt = f"Route the conversation to one of: {', '.join(options)}. Respond ONLY with the name of the agent or FINISH."
    
    messages = [HumanMessage(content=system_prompt + "\n\n" + prompt)] + state["messages"]
    response = llm.invoke(messages)
    
    # Simple parsing logic
    goto = response.content.strip()
    if goto not in options:
        goto = "GeneralAgent" # Fallback
        
    return {"next": goto}

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("CodingAgent", coding_node)
workflow.add_node("MathAgent", math_node)
workflow.add_node("KnowledgeAgent", knowledge_node)
workflow.add_node("GeneralAgent", general_node)

# Add edges - every agent reports back to the supervisor
workflow.add_edge("CodingAgent", "Supervisor")
workflow.add_edge("MathAgent", "Supervisor")
workflow.add_edge("KnowledgeAgent", "Supervisor")
workflow.add_edge("GeneralAgent", "Supervisor")

# The supervisor decides where to go next
conditional_map = {name: name for name in ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]}
conditional_map["FINISH"] = END

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    conditional_map
)

workflow.set_entry_point("Supervisor")

# Compile
graph = workflow.compile()

def run_orchestrator(query: str, langfuse_handler: CallbackHandler = None):
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    inputs = {"messages": [HumanMessage(content=query)]}
    final_state = graph.invoke(inputs, config=config)
    return final_state["messages"][-1].content
