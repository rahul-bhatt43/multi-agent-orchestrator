import os
import uuid
import time
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

# Load environment variables
load_dotenv()

try:
    from langfuse.callback import CallbackHandler
except ImportError:
    CallbackHandler = None

from orchestrator import run_orchestrator

from rich.theme import Theme
from prompt_toolkit import PromptSession
import re
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PtStyle

# Load environment variables
load_dotenv()

try:
    from langfuse.callback import CallbackHandler
except ImportError:
    CallbackHandler = None

from orchestrator import run_orchestrator

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
COMMANDS = ["/new", "/switch", "/list", "/exit", "/help"]
# Use a compiled regex pattern for matching commands
command_completer = WordCompleter(COMMANDS, pattern=re.compile(r'^/.*'))

def display_welcome():
    welcome_text = """
# Multi-Agent Orchestration System
Welcome to your premium AI assistant! This system orchestrates multiple specialized agents 
(Coding, Math, Knowledge, General) to provide accurate and context-aware responses.

**Commands:**
- `/new`: Start a fresh session
- `/switch <id>`: Switch conversations
- `/list`: See active sessions
- `/help`: Show this help message
- `/exit`: Quit the application

*Type [bold blue]/[/] to see suggestions.*
"""
    console.print(Panel(Markdown(welcome_text), border_style="blue", title="[bold blue]System Booted[/]"))

def main():
    display_welcome()

    langfuse_handler = None
    if CallbackHandler is not None:
        langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

    sessions = {"default": str(uuid.uuid4())}
    current_session = "default"
    
    # Initialize prompt_toolkit session
    pt_session = PromptSession(completer=command_completer)
    
    while True:
        try:
            # Use prompt_toolkit for suggestions
            user_input = pt_session.prompt(f"[{current_session}] User: ").strip()
            
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
                    console.print(f"[bold info]Started new session:[/] [session]{current_session}[/]")
                    continue
                    
                elif cmd == "/switch":
                    if len(cmd_parts) > 1:
                        target = cmd_parts[1]
                        if target in sessions:
                            current_session = target
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

            # Run with status indicator
            with console.status("[bold magenta]Orchestrating agents...[/]", spinner="dots"):
                response = run_orchestrator(
                    user_input, 
                    thread_id=sessions[current_session], 
                    langfuse_handler=langfuse_handler
                )
            
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
