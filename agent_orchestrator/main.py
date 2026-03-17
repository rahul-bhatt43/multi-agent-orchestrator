import os
import uuid
import time
import re
from dotenv import load_dotenv
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.status import Status
from rich.theme import Theme

from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.key_binding import KeyBindings

# Load environment variables
load_dotenv()

try:
    from langfuse.langchain import CallbackHandler
except ImportError:
    CallbackHandler = None

from .orchestrator import run_orchestrator

# Custom theme for a premium feel

# Custom theme for a premium feel
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "user": "bold green",
    "assistant": "bold magenta",
    "session": "bold blue"
})

console = Console(theme=custom_theme)

# Commands for auto-completion
COMMANDS = ["/new", "/switch", "/list", "/exit", "/help", "/add"]
command_completer = WordCompleter(COMMANDS, ignore_case=True, sentence=True)

def display_welcome():
    welcome_text = """
# Multi-Agent Orchestration System
Welcome to your premium AI assistant! This system orchestrates multiple specialized agents 
(Coding, Math, Knowledge, General) to provide accurate and context-aware responses.

**Commands:**
- `/new`: Start a fresh session
- `/switch <id>`: Switch conversations
- `/list`: See active sessions
- `/add <file>`: Give file context to agents
- `/help`: Show this help message
- `/exit`: Quit the application

*Type [bold blue]/[/] to see suggestions.*
"""
    console.print(Panel(Markdown(welcome_text), border_style="blue", title="[bold blue]System Booted[/]"))

def read_file_content(file_path: str) -> Optional[str]:
    """Reads a file and returns its content or None if error."""
    try:
        # Resolve relative path based on current working directory
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            console.print(f"[error]Error: File '{file_path}' not found.[/]")
            return None
        
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        console.print(f"[error]Error reading file:[/] {e}")
        return None

def validate_environment():
    """Checks if required environment variables are set."""
    # OpenAI Check
    if not os.getenv("OPENAI_API_KEY"):
        error_msg = """
# Configuration Missing
The [bold yellow]OPENAI_API_KEY[/] is not set.

**How to fix:**
- **Option 1 (Bash/Gerrit/Zsh):** `export OPENAI_API_KEY="your-key"`
- **Option 2 (PowerShell):** `$env:OPENAI_API_KEY = "your-key"`
- **Option 3 (Interactive):** I can ask you for it now.
"""
        console.print(Panel(Markdown(error_msg), border_style="red", title="[bold red]API Key Missing[/]"))
        
        choice = Prompt.ask("Would you like to enter your API key interactively now? (y/n)", choices=["y", "n"], default="y")
        if choice == "y":
            api_key = Prompt.ask("Enter your OpenAI API Key", password=True)
            if api_key.startswith("sk-"):
                os.environ["OPENAI_API_KEY"] = api_key
                if Prompt.ask("Save to .env?", choices=["y", "n"], default="y") == "y":
                    with open(".env", "a") as f: f.write(f"\nOPENAI_API_KEY={api_key}\n")
            else:
                console.print("[bold red]Error:[/] Invalid key format.")
                return False
        else:
            return False

    # Semantic Memory Check
    if not os.getenv("SEMANTIC_MEMORY_ENABLED"):
        choice = Prompt.ask("Enable Long-term Semantic Memory (Pinecone)? (y/n)", choices=["y", "n"], default="n")
        if choice == "y":
            os.environ["SEMANTIC_MEMORY_ENABLED"] = "true"
            # Check for Pinecone Keys
            for key in ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]:
                if not os.getenv(key):
                    val = Prompt.ask(f"Enter your {key.replace('_', ' ').title()}")
                    os.environ[key] = val
                    if Prompt.ask(f"Save {key} to .env?", choices=["y", "n"], default="y") == "y":
                        with open(".env", "a") as f: f.write(f"\n{key}={val}\n")
            with open(".env", "a") as f: f.write("\nSEMANTIC_MEMORY_ENABLED=true\n")
        else:
            os.environ["SEMANTIC_MEMORY_ENABLED"] = "false"
            if Prompt.ask("Disable this prompt in future? (y/n)", choices=["y", "n"], default="y") == "y":
                with open(".env", "a") as f: f.write("\nSEMANTIC_MEMORY_ENABLED=false\n")
    
    # Langfuse Check
    if not os.getenv("LANGFUSE_ENABLED"):
        choice = Prompt.ask("Enable Langfuse Tracing? (y/n)", choices=["y", "n"], default="n")
        if choice == "y":
            os.environ["LANGFUSE_ENABLED"] = "true"
            # Check for Langfuse Keys
            for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]:
                if not os.getenv(key):
                    val = Prompt.ask(f"Enter your {key.replace('_', ' ').title()}")
                    os.environ[key] = val
                    if Prompt.ask(f"Save {key} to .env?", choices=["y", "n"], default="y") == "y":
                        with open(".env", "a") as f: f.write(f"\n{key}={val}\n")
            
            if not os.getenv("LANGFUSE_HOST"):
                host = Prompt.ask("Enter Langfuse Host", default="https://cloud.langfuse.com")
                os.environ["LANGFUSE_HOST"] = host
                if Prompt.ask("Save HOST to .env?", choices=["y", "n"], default="y") == "y":
                    with open(".env", "a") as f: f.write(f"\nLANGFUSE_HOST={host}\n")
            
            with open(".env", "a") as f: f.write("\nLANGFUSE_ENABLED=true\n")
        else:
            os.environ["LANGFUSE_ENABLED"] = "false"
            if Prompt.ask("Disable Langfuse prompt in future? (y/n)", choices=["y", "n"], default="y") == "y":
                with open(".env", "a") as f: f.write("\nLANGFUSE_ENABLED=false\n")
    
    return True

def main():
    # Load environment variables (relevant when run as a tool)
    load_dotenv()
    
    if not validate_environment():
        return

    display_welcome()

    # --- Langfuse Initialization ---
    if CallbackHandler is not None and os.getenv("LANGFUSE_ENABLED") == "true":
        try:
            # Langfuse v2+ standard: picks up keys/host from environment variables
            langfuse_handler = CallbackHandler()
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            console.print(f"[bold green]✔[/] Langfuse Tracing Enabled ([dim]{host}[/])")
        except Exception as e:
            console.print(f"[bold red]✘[/] Langfuse Init Error: {e}")
    else:
        console.print("[dim]Langfuse Tracing: Disabled[/]")
    # ---------------------------

    sessions = {"default": str(uuid.uuid4())}
    current_session = "default"
    
    # Custom key bindings for multi-line support
    kb = KeyBindings()
    
    @kb.add('enter')
    def _(event):
        """Submit on Enter (if not just a newline)"""
        # If the user is typing a command, we might want to just submit.
        # But for general chat, Enter should submit.
        event.current_buffer.validate_and_handle()

    @kb.add('escape', 'enter')
    def _(event):
        """Insert a newline on Alt+Enter (Escape + Enter)"""
        event.current_buffer.insert_text('\n')

    # Initialize prompt_toolkit session with better completion style
    pt_session = PromptSession(
        completer=command_completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=kb
        # enable_system_clipboard=True
    )
    
    while True:
        try:
            # Use prompt_toolkit for suggestions
            # Use prompt_toolkit for multi-line and paste support
            user_input = pt_session.prompt(
                f"[{current_session}] User: ",
                multiline=True
            ).strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("[bold red]Shutting down... Goodbye![/]")
                break
            
            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()
                
                if cmd == "/new":
                    new_id = f"session_{len(sessions)}"
                    sessions[new_id] = str(uuid.uuid4())
                    current_session = new_id
                    os.system('cls' if os.name == 'nt' else 'clear')
                    display_welcome()
                    console.print(f"[bold info]Started new session:[/] [session]{current_session}[/]")
                    continue
                    
                elif cmd == "/switch":
                    if len(cmd_parts) > 1:
                        target = cmd_parts[1]
                        if target in sessions:
                            current_session = target
                            os.system('cls' if os.name == 'nt' else 'clear')
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
                            context_msg = f"--- BEGIN FILE CONTEXT: {file_path} ---\n{content}\n--- END FILE CONTEXT ---"
                            # Inject context into orchestrator as a background "context provide" message
                            with console.status(f"[bold info]Adding {file_path} to context...[/]"):
                                # We send it as a "hidden" query that the agent acknowledges
                                run_orchestrator(
                                    f"I am providing the content of '{file_path}' as context. Please acknowledge you've received it.\n\n{context_msg}",
                                    thread_id=sessions[current_session],
                                    langfuse_handler=langfuse_handler
                                )
                            console.print(f"[bold info]File context added:[/] [session]{file_path}[/]")
                        continue
                    else:
                        console.print("[warning]Usage: /add <file_path>[/]")
                        continue

                elif cmd == "/help":
                    display_welcome()
                    continue
                
                else:
                    # Fallback for unknown commands
                    console.print(Panel(
                        f"[warning]Unknown command:[/] [bold]{cmd}[/]\n\n"
                        "Available commands: [bold]" + ", ".join(COMMANDS) + "[/]\n"
                        "If you wanted to ask a question, please don't start with [bold yellow]/[/].",
                        title="[error]Command Error[/]",
                        border_style="red"
                    ))
                    continue

            # Run the orchestrator (status handles by internal nodes)
            response = run_orchestrator(
                user_input, 
                thread_id=sessions[current_session], 
                langfuse_handler=langfuse_handler
            )
            # Flush traces to ensure they are sent (using safe check for different library versions)
            if langfuse_handler:
                try:
                    if hasattr(langfuse_handler, "flush"):
                        langfuse_handler.flush()
                    elif hasattr(langfuse_handler, "_langfuse_client") and hasattr(langfuse_handler._langfuse_client, "flush"):
                        langfuse_handler._langfuse_client.flush()
                except:
                    pass
            
            # Formatted Assistant Output
            console.print(Panel(
                Markdown(response), 
                title=f"[bold magenta]Assistant ({current_session})[/]", 
                border_style="magenta",
                padding=(1, 2)
            ))
            
        except KeyboardInterrupt:
            continue # Allow ctrl+c to clear line
        except EOFError:
            console.print("\n[bold red]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[error]Error:[/] {e}")

if __name__ == "__main__":
    main()
