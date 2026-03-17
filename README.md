# 🤖 Agent Orchestrator

A premium, multi-agent orchestration system with optional long-term semantic memory and a professional CLI. This tool allows you to interact with specialized AI agents that can code, solve math problems, search knowledge bases, and more—all while maintaining context across sessions.

## ✨ Features

- **Multi-Agent Flow**: Orchestrated by a Supervisor agent using LangGraph.
- **Optional Semantic Memory**: Toggle long-term context awareness via Pinecone. Store and retrieve memories across different project sessions.
- **Premium CLI**: Built with `rich` and `prompt_toolkit` for a beautiful, interactive experience with command suggestions.
- **File Context Support**: Add external files to any session using `/add <path>`.
- **Global Access**: Install as a global tool and run `orch` from any directory.
- **Interactive Onboarding**: Guides you through setting up OpenAI and Pinecone keys securely within the terminal.

## 🚀 Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/agent-orchestrator.git
   cd agent-orchestrator
   ```

2. **Install globally as a tool**:
   ```bash
   uv tool install . --force
   ```

3. **Run from anywhere**:
   ```bash
   orch
   ```

### Local Development
If you want to run the project locally without installing it globally:
```bash
uv run python -m agent_orchestrator.main
```

## 🛠️ Project Structure

```text
agent-orchestrator/
├── agent_orchestrator/       # Core package
│   ├── agents/               # Specialized agent definitions
│   │   ├── base.py           # Base agent logic & Memory integration
│   │   ├── coding.py         # coding expert
│   │   ├── math.py           # math specialist
│   │   └── ...
│   ├── main.py               # CLI entry point & UI logic
│   └── orchestrator.py       # LangGraph orchestration & supervisor
├── pyproject.toml            # Package metadata & dependencies
└── README.md                 # You are here
```

## 🧠 How it Works

1. **User Input**: You type a request or a command (like `/add`).
2. **Supervisor**: A central "Source Control Agent" analyzes your request.
3. **Routing**: The supervisor routes the task to the best specialized agent (Coding, Math, etc.).
4. **Memory Management**:
   - **Short-term**: Managed by LangGraph's `MemorySaver` (maintains context within a single session).
   - **Long-term (Optional)**: If enabled, interactions are vectorized and stored in **Pinecone**.
   - **Privacy First**: You can choose to disable Semantic Memory at startup for a purely session-based, private interaction.

## ⌨️ Commands

| Command | Description |
| :--- | :--- |
| `/new` | Start a fresh conversational session |
| `/switch <id>` | Switch between active sessions |
| `/list` | List all active sessions and their IDs |
| `/add <path>` | Feed a file's content into the agent's context |
| `/help` | Display the welcome and help menu |
| `/exit` | Safely shutdown the application |

## ⚠️ Troubleshooting

### Installation: "Access is denied"
If you see an `Access is denied` error during `uv tool install`, it means an existing instance of `orch` or Python is locking the files.
- **Fix**: Close all active terminal sessions running the tool and try again. 
- **PowerShell Fix**: `taskkill /F /IM python.exe /T` (Warning: This stops all running Python processes).

## 📝 License
MIT
