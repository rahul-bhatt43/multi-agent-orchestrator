import os
import uuid
from dotenv import load_dotenv
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.key_binding import KeyBindings

# ─────────────────────────────────────────────────────────────────────────────
# Credential loading: .orch first, then fall back to .env
# ─────────────────────────────────────────────────────────────────────────────
ORCH_FILE = ".orch"


def _load_credentials():
    """Load .orch then .env (if present). .orch is the preferred store."""
    if os.path.exists(ORCH_FILE):
        load_dotenv(ORCH_FILE, override=True)
    # Also load .env so existing setups still work (OPENAI_API_KEY etc.)
    load_dotenv(override=False)


def _save_to_orch(key: str, value: str):
    """Append or update a key=value line in the .orch file."""
    lines = []
    if os.path.exists(ORCH_FILE):
        with open(ORCH_FILE, "r") as f:
            lines = f.readlines()

    # Remove existing line with this key (if any)
    lines = [l for l in lines if not l.strip().startswith(f"{key}=")]
    lines.append(f"{key}={value}\n")

    with open(ORCH_FILE, "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────

try:
    from langfuse.langchain import CallbackHandler as LangfuseHandler
except ImportError:
    LangfuseHandler = None

from .orchestrator import run_orchestrator

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "user": "bold green",
    "assistant": "bold magenta",
    "session": "bold blue",
})

console = Console(theme=custom_theme)

COMMANDS = ["/new", "/switch", "/list", "/exit", "/help", "/add", "/abort"]
command_completer = WordCompleter(COMMANDS, ignore_case=True, sentence=True)


def display_welcome():
    welcome_text = """
# Multi-Agent Orchestration System
Welcome to your AI assistant. Specialized agents for Coding, Math, Knowledge, and General tasks.

**Commands:**
- `/new`: Start a fresh session
- `/switch <id>`: Switch conversations
- `/list`: See active sessions
- `/add <file>`: Give file context to agents
- `/abort`: Cancel a running or stuck agent immediately
- `/help`: Show this help message
- `/exit`: Quit the application

**Escape hatch:** Press `Ctrl+C` at any time (including during a permission prompt) to abort the current agent task.

*Type [bold blue]/[/] to see suggestions.*
"""
    console.print(Panel(Markdown(welcome_text), border_style="blue", title="[bold blue]System Booted[/]"))


def read_file_content(file_path: str) -> Optional[str]:
    """Reads a file and returns its content or None if error."""
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            console.print(f"[error]Error: File '{file_path}' not found.[/]")
            return None
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        console.print(f"[error]Error reading file:[/] {e}")
        return None


def validate_environment():
    """
    Checks required environment variables.
    - OpenAI key: required, prompts if missing.
    - Langfuse: fully optional, silent unless LANGFUSE_ENABLED=true is set but keys are missing.
    - Pinecone: fully optional, no prompts.
    All saves go to .orch (not .env).
    """
    # ── OpenAI (required) ────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = """
# Configuration Missing
The [bold yellow]OPENAI_API_KEY[/] is not set.

**How to fix:**
- Add `OPENAI_API_KEY=sk-...` to a `.orch` file in your project directory, **or**
- Set it as an environment variable first.
"""
        console.print(Panel(Markdown(error_msg), border_style="red", title="[bold red]API Key Missing[/]"))

        choice = Prompt.ask("Enter your OpenAI API Key interactively now? (y/n)", choices=["y", "n"], default="y")
        if choice == "y":
            api_key = Prompt.ask("OpenAI API Key", password=True)
            if api_key.startswith("sk-"):
                os.environ["OPENAI_API_KEY"] = api_key
                if Prompt.ask("Save to .orch?", choices=["y", "n"], default="y") == "y":
                    _save_to_orch("OPENAI_API_KEY", api_key)
                    console.print(f"[bold green]✔[/] Saved to [bold]{ORCH_FILE}[/]")
            else:
                console.print("[bold red]Error:[/] Invalid key format (must start with sk-).")
                return False
        else:
            return False

    # ── Pinecone (optional — no prompts) ─────────────────────────────────────
    # Already handled gracefully in base.py — just normalise the env var
    if not os.getenv("SEMANTIC_MEMORY_ENABLED"):
        os.environ["SEMANTIC_MEMORY_ENABLED"] = "false"

    # ── Langfuse (optional — silent unless already enabled) ──────────────────
    langfuse_enabled_raw = os.getenv("LANGFUSE_ENABLED", "").lower()
    if langfuse_enabled_raw == "true":
        # User has opted in — verify keys are present
        missing_langfuse = [k for k in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"] if not os.getenv(k)]
        if missing_langfuse:
            console.print(f"[warning]⚠ LANGFUSE_ENABLED=true but keys are missing: {', '.join(missing_langfuse)}[/]")
            console.print("[dim]Langfuse tracing will be disabled for this session.[/]")
            os.environ["LANGFUSE_ENABLED"] = "false"
    else:
        # Not configured → silently disable, no prompt
        os.environ["LANGFUSE_ENABLED"] = "false"

    return True


def main():
    _load_credentials()

    if not validate_environment():
        return

    display_welcome()

    # ── Langfuse initialization ───────────────────────────────────────────────
    langfuse_handler = None
    if LangfuseHandler is not None and os.getenv("LANGFUSE_ENABLED") == "true":
        try:
            langfuse_handler = LangfuseHandler()
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            console.print(f"[bold green]✔[/] Langfuse Tracing Enabled ([dim]{host}[/])")
        except Exception as e:
            console.print(f"[bold red]✘[/] Langfuse Init Error: {e}")
            langfuse_handler = None
    else:
        console.print("[dim]Langfuse Tracing: Disabled[/]")

    sessions = {"default": str(uuid.uuid4())}
    current_session = "default"

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    pt_session = PromptSession(
        completer=command_completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=kb,
    )

    session_auto_approve = False

    # ── Permission handler ────────────────────────────────────────────────────

    def handle_permission_request(tool_name: str, tool_args: dict) -> bool:
        """Shows the proposed change and asks Allow / Decline / Allow Always."""
        nonlocal session_auto_approve
        if session_auto_approve:
            return True

        console.print("\n" + "─" * 40 + " [bold yellow]Review & Approve[/] " + "─" * 40)

        if tool_name == "plan_code_changes":
            plan_text = tool_args.get("plan", "")
            console.print(Panel(
                Markdown(plan_text),
                title="[bold cyan]📋 Proposed Plan[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            ))

        elif tool_name == "write_project_file":
            file_path = tool_args.get("file_path", "unknown")
            new_content = tool_args.get("content", "")

            # Compute unified diff
            from .agents.edit_tools import compute_diff
            diff_text = compute_diff(file_path, new_content)

            console.print(f"[bold info]File:[/] [session]{file_path}[/]")

            if diff_text:
                # Show a coloured diff
                console.print(Panel(
                    Syntax(diff_text, "diff", theme="monokai", line_numbers=False),
                    title=f"[bold green]📝 File Write — Diff View[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                ))
            else:
                # New file — show syntax-highlighted content
                ext = os.path.splitext(file_path)[1].lstrip(".") or "text"
                console.print(Panel(
                    Syntax(new_content, ext, theme="monokai", line_numbers=True),
                    title=f"[bold green]📝 New File: {file_path}[/bold green]",
                    border_style="green",
                    padding=(0, 1),
                ))

        else:
            # Generic fallback
            console.print(Panel(
                Text(str(tool_args), style="dim"),
                title=f"[bold yellow]Tool: {tool_name}[/bold yellow]",
                border_style="yellow",
            ))

        choice = Prompt.ask(
            "\n[bold yellow]  Allow[/] (y) / [bold red]Decline[/] (n) / [bold green]Allow Always[/] (a)",
            choices=["y", "n", "a"],
            default="y",
        )

        if choice == "a":
            session_auto_approve = True
            console.print("[bold green]✔ 'Allow Always' enabled for this session.[/]\n")
            return True
        elif choice == "y":
            console.print("[bold green]✔ Approved.[/]\n")
            return True
        else:
            console.print("[bold red]✘ Declined.[/]\n")
            return False

    # ── Main loop ─────────────────────────────────────────────────────────────

    while True:
        try:
            user_input = pt_session.prompt(
                f"[{current_session}] You: ",
                multiline=True,
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[bold red]Shutting down... Goodbye![/]")
                break

            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd == "/new":
                    new_id = f"session_{len(sessions)}"
                    sessions[new_id] = str(uuid.uuid4())
                    current_session = new_id
                    os.system("cls" if os.name == "nt" else "clear")
                    display_welcome()
                    console.print(f"[bold info]Started new session:[/] [session]{current_session}[/]")
                    continue

                elif cmd == "/switch":
                    if len(cmd_parts) > 1:
                        target = cmd_parts[1]
                        if target in sessions:
                            current_session = target
                            os.system("cls" if os.name == "nt" else "clear")
                            display_welcome()
                            console.print(f"[bold info]Switched to session:[/] [session]{current_session}[/]")
                        else:
                            console.print(f"[error]Error: Session '{target}' not found.[/]")
                    else:
                        console.print("[warning]Usage: /switch <session_name>[/]")
                    continue

                elif cmd == "/list":
                    table = Table(title="Active Sessions", border_style="blue")
                    table.add_column("Session Name", style="session")
                    table.add_column("Status", justify="center")
                    table.add_column("Unique ID", style="dim")
                    for s in sessions:
                        marker = "[bold green]Active[/]" if s == current_session else ""
                        table.add_row(s, marker, sessions[s])
                    console.print(table)
                    continue

                elif cmd == "/add":
                    if len(cmd_parts) > 1:
                        file_path = cmd_parts[1]
                        content = read_file_content(file_path)
                        if content:
                            context_msg = (
                                f"--- BEGIN FILE CONTEXT: {file_path} ---\n"
                                f"{content}\n"
                                f"--- END FILE CONTEXT ---"
                            )
                            with console.status(f"[bold info]Adding {file_path} to context...[/]"):
                                run_orchestrator(
                                    f"I am providing the content of '{file_path}' as context. "
                                    f"Please acknowledge you've received it.\n\n{context_msg}",
                                    thread_id=sessions[current_session],
                                    langfuse_handler=langfuse_handler,
                                )
                            console.print(f"[bold info]File context added:[/] [session]{file_path}[/]")
                        continue
                    else:
                        console.print("[warning]Usage: /add <file_path>[/]")
                        continue

                elif cmd == "/help":
                    display_welcome()
                    continue

                elif cmd == "/abort":
                    console.print("[bold yellow]⚠ Nothing is currently running. /abort only works while an agent is processing.[/]")
                    continue

                else:
                    console.print(Panel(
                        f"[warning]Unknown command:[/] [bold]{cmd}[/]\n\n"
                        "Available commands: [bold]" + ", ".join(COMMANDS) + "[/]",
                        title="[error]Command Error[/]",
                        border_style="red",
                    ))
                    continue

            try:
                response = run_orchestrator(
                    user_input,
                    thread_id=sessions[current_session],
                    langfuse_handler=langfuse_handler,
                    on_tool_call=handle_permission_request,
                )
            except KeyboardInterrupt:
                console.print("\n[bold yellow]⚠ Agent aborted by user (Ctrl+C).[/]")
                continue

            # Flush Langfuse traces
            if langfuse_handler:
                try:
                    if hasattr(langfuse_handler, "flush"):
                        langfuse_handler.flush()
                    elif hasattr(langfuse_handler, "_langfuse_client"):
                        langfuse_handler._langfuse_client.flush()
                except Exception:
                    pass

            console.print(Panel(
                Markdown(response),
                title=f"[bold magenta]Assistant ({current_session})[/]",
                border_style="magenta",
                padding=(1, 2),
            ))

        except KeyboardInterrupt:
            # Ctrl+C at the prompt — just clear the line and continue
            console.print("[dim](Ctrl+C — type /exit to quit)[/]")
            continue
        except EOFError:
            console.print("\n[bold red]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[error]Error:[/] {e}")


if __name__ == "__main__":
    main()
