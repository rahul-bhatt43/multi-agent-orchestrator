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
        # Identify if this is a new turn (no AIMessage yet for the current HumanMessage)
        # In our graph, the last message is either the initial HumanMessage or the result of an agent.
        messages = state["messages"]
        last_message = messages[-1]
        is_new_turn = isinstance(last_message, HumanMessage)
        
        system_prompt = (
            "You are the Multi-Agent Supervisor. Your goal is to coordinate specialized agents "
            "to answer the user's request. \n"
            "Current Agents: \n"
            "- CodingAgent: Handles code, scripts, and technical tasks.\n"
            "- MathAgent: Handles complex calculations and math problems requiring a calculator.\n"
            "- KnowledgeAgent: ONLY for retrieving private information from the user's uploaded files (/add) or specific personal details mentioned in the past.\n"
            "- GeneralAgent: Handles general conversation, common world knowledge (science, history, basic theory), and general logic. If the question is about a general topic like 'Pythagorean Theorem' and NOT a specific file the user added, pick this agent.\n\n"
            "Rules:\n"
            "1. If an agent (AIMessage) has already provided a complete answer to the LATEST user request, respond with 'FINISH'.\n"
            "2. If the user just spoke and no agent has responded yet, you MUST route to one of the agents. Do NOT respond with 'FINISH' immediately.\n"
            "3. If the user is asking about themselves, past info, or something previously mentioned, route to 'KnowledgeAgent' first.\n"
            "4. Respond ONLY with the name of the agent to call next, or 'FINISH' if the task is done."
        )
        members = ["CodingAgent", "MathAgent", "KnowledgeAgent", "GeneralAgent"]
        options = ["FINISH"] + members
        
        prompt = f"Based on the conversation, route to one of ({', '.join(options)})."
        
        # Prepare messages for the supervisor
        supervisor_messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)] + messages
        response = llm.invoke(supervisor_messages, config=config)
        
        goto = response.content.strip().replace("'", "").replace('"', "")
        if goto not in options:
            goto = "GeneralAgent"
        
        # STRICTOR FINISH LOGIC:
        # If the last message is an AIMessage, we generally want to FINISH.
        # Only continue if the AI response was just a routing step or needs more data.
        if not is_new_turn and goto != "FINISH":
            # Check if the last AI message was actually a response or just an error/thinking
            # For simplicity, if an agent spoke, we finish unless the user query was complex.
            # GPT-4o-mini sometimes over-routes; let's force a finish if it tries to route to the SAME agent again.
            last_agent = getattr(last_message, "name", None)
            if goto == last_agent:
                goto = "FINISH"
        
        # Prevent premature FINISH on a new turn
        if goto == "FINISH" and is_new_turn:
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
    # LangGraph with memory allows us to get the initial state length
    initial_messages_count = len(graph.get_state(config).values.get("messages", []))
    
    inputs = {"messages": [HumanMessage(content=query)]}
    final_state = graph.invoke(inputs, config=config)
    
    # Get all messages in the final state
    all_messages = final_state["messages"]
    
    # New messages are those added after the initial ones + the one we just injected
    # our query is at index initial_messages_count
    new_messages = all_messages[initial_messages_count + 1:]
    
    # Filter for AI responses in the new messages
    agent_responses = [m for m in new_messages if isinstance(m, AIMessage)]
    
    if agent_responses:
        # Join multiple responses if the supervisor routed to multiple agents sequentially
        assistant_response = "\n\n".join([m.content for m in agent_responses])
    else:
        # Check if the last message in the whole list is an AI message from a previous turn
        # that addressed the current query (unlikely with FINISH logic but good to have)
        if all_messages and isinstance(all_messages[-1], AIMessage):
             assistant_response = all_messages[-1].content
        else:
            assistant_response = "I've acknowledged your request. How else can I help?"
    
    # Store the interaction for semantic memory
    store_memory(query, assistant_response, thread_id)
    
    return assistant_response
