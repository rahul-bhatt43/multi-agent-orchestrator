import os
from rich import print
from datetime import datetime
from typing import List, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool


def _pinecone_enabled() -> bool:
    return (
        os.getenv("SEMANTIC_MEMORY_ENABLED") == "true"
        and bool(os.getenv("PINECONE_INDEX_NAME"))
        and bool(os.getenv("PINECONE_API_KEY"))
    )


def store_memory(query: str, response: str, thread_id: str):
    """Stores a query-response pair into Pinecone for long-term memory."""
    if not _pinecone_enabled():
        return
    try:
        from langchain_pinecone import PineconeVectorStore
        index_name = os.getenv("PINECONE_INDEX_NAME")
        api_key = os.getenv("PINECONE_API_KEY")
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)
        doc = Document(
            page_content=f"User: {query}\nAssistant: {response}",
            metadata={"thread_id": thread_id, "timestamp": datetime.now().isoformat(), "type": "chat_history"},
        )
        vectorstore.add_documents([doc])
    except Exception as e:
        print(f"[dim]DEBUG: Error storing memory: {e}[/]")


def retrieve_memories(query: str, k: int = 5) -> str:
    """Retrieves relevant past interactions from Pinecone."""
    if not _pinecone_enabled():
        return ""
    try:
        from langchain_pinecone import PineconeVectorStore
        index_name = os.getenv("PINECONE_INDEX_NAME")
        api_key = os.getenv("PINECONE_API_KEY")
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return ""
        memory_text = "\n---\n".join([doc.page_content for doc in docs])
        return f"\nRelevant past interactions:\n{memory_text}\n"
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Agent loop sentinel: track (tool_name, frozen_args) to detect identical retries
# ──────────────────────────────────────────────────────────────────────────────

def _freeze_args(args: dict) -> str:
    """Return a stable string representation of tool args for loop detection."""
    try:
        import json
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)


def create_agent(llm: ChatOpenAI, tools: List[Union[BaseTool, callable]], system_prompt: str):
    """
    Returns an Agent instance that handles system prompt and tool execution.
    """
    class Agent:
        def __init__(self, llm, tools, system_prompt):
            self.llm = llm
            self.tools = tools
            self.system_prompt = system_prompt

        def agent_node(self, inputs: dict, config: dict = None):
            messages = inputs.get("messages", [])
            user_query = messages[-1].content if messages else ""

            # Retrieve long-term semantic context (only if Pinecone is configured)
            semantic_context = retrieve_memories(user_query)
            full_system_prompt = self.system_prompt
            if semantic_context:
                full_system_prompt += f"\n\n{semantic_context}\nUse the above context if relevant."

            combined_messages = [SystemMessage(content=full_system_prompt)] + list(messages)

            try:
                llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

                MAX_LOOPS = 15
                loop_count = 0
                # Track last call to detect stuck loops: (tool_name, frozen_args)
                last_call_signature: str | None = None
                # Track whether plan_code_changes was approved in this agent run
                _plan_approved: bool = False

                while loop_count < MAX_LOOPS:
                    loop_count += 1
                    response = llm_with_tools.invoke(combined_messages, config=config)

                    # ── Terminal: no more tool calls → return final answer ──
                    if not response.tool_calls:
                        thread_id = "default"
                        if config and "configurable" in config:
                            thread_id = config["configurable"].get("thread_id", "default")
                        store_memory(user_query, response.content, thread_id)
                        return {"messages": [response]}

                    # ── Guard: plan must come before write ──
                    tool_names_in_round = [tc["name"] for tc in response.tool_calls]
                    if "write_project_file" in tool_names_in_round and "plan_code_changes" not in tool_names_in_round:
                        if not _plan_approved:
                            combined_messages.append(response)
                            for tc in response.tool_calls:
                                if tc["name"] == "write_project_file":
                                    combined_messages.append(ToolMessage(
                                        content=(
                                            "Error: You MUST call `plan_code_changes` before `write_project_file`. "
                                            "Please call `plan_code_changes` first with a clear description of what "
                                            "you intend to change, then call `write_project_file`."
                                        ),
                                        tool_call_id=tc["id"],
                                    ))
                            continue

                    combined_messages.append(response)

                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]

                        # ── Loop detection: same tool + same args twice in a row ──
                        call_sig = f"{tool_name}::{_freeze_args(tool_args)}"
                        if call_sig == last_call_signature:
                            combined_messages.append(ToolMessage(
                                content=(
                                    f"Error: Detected an exact retry of the same '{tool_name}' call. "
                                    "Retrying the same action will not produce a different result. "
                                    "Please re-read the relevant file and think about a different approach, "
                                    "or report the issue to the user."
                                ),
                                tool_call_id=tool_call["id"],
                            ))
                            continue
                        last_call_signature = call_sig

                        permission_callback = None
                        if config and "configurable" in config:
                            permission_callback = config["configurable"].get("on_tool_call")

                        SENSITIVE_TOOLS = {"write_project_file", "plan_code_changes"}

                        tool_fn = next(
                            (t for t in self.tools if
                             getattr(t, "name", None) == tool_name or
                             getattr(t, "__name__", None) == tool_name),
                            None,
                        )

                        if tool_fn:
                            try:
                                # Permission gate for sensitive tools
                                if permission_callback and tool_name in SENSITIVE_TOOLS:
                                    approved = permission_callback(tool_name, tool_args)
                                    if not approved:
                                        print(f"\n[bold red]✘[/] User declined '{tool_name}'.")
                                        combined_messages.append(ToolMessage(
                                            content=(
                                                "The user declined this action. Do NOT retry it. "
                                                "Ask the user what they would like you to do instead."
                                            ),
                                            tool_call_id=tool_call["id"],
                                        ))
                                        continue

                                result = tool_fn.invoke(tool_args) if hasattr(tool_fn, "invoke") else tool_fn(**tool_args)

                                # Mark plan as approved so the write guard passes
                                if tool_name == "plan_code_changes":
                                    _plan_approved = True

                                if tool_name in SENSITIVE_TOOLS:
                                    success = not str(result).lower().startswith("error")
                                    color = "green" if success else "red"
                                    print(f"\n[{color}]▶ [Tool Result]: {result}[/]")

                                combined_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

                            except Exception as e:
                                combined_messages.append(ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"]))
                        else:
                            combined_messages.append(ToolMessage(
                                content=f"Tool '{tool_name}' not found.", tool_call_id=tool_call["id"]
                            ))

                return {"messages": [AIMessage(
                    content=(
                        "I've reached the maximum number of steps for this task. "
                        "This may indicate the request was too complex for a single turn. "
                        "Please let me know how you'd like to proceed."
                    )
                )]}

            except Exception as e:
                print(f"[dim]DEBUG: Error in agent: {e}[/]")
                return {"messages": [AIMessage(content=f"An error occurred: {e}")]}

        def invoke(self, inputs: dict, config: dict = None):
            return self.agent_node(inputs, config=config)

    return Agent(llm, tools, system_prompt)