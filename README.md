# 🤖 Agent Orchestrator

A premium, multi-agent orchestration system with long-term semantic memory and a professional CLI. This tool allows you to interact with specialized AI agents that can code, solve math problems, search knowledge bases, and more—all while maintaining context across sessions.

## ✨ Features

- **Multi-Agent Flow**: Orchestrated by a Supervisor agent using LangGraph.
- **Semantic Memory**: Integrated with Pinecone to remember conversations across sessions.
- **Premium CLI**: Built with `rich` and `prompt_toolkit` for a beautiful, interactive experience with command suggestions.
- **File Context Support**: Add external files to any session using `/add <path>`.
- **Global Access**: Install as a global tool and run `orch` from any directory.
- **Interactive Onboarding**: Missing API keys? No problem. The tool guides you through setup interactively.

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
│   │   ├── base.py           # Base agent logic & Pinecone integration
│   │   ├── coding.py         # coding expert
│   │   ├── math.py           # math specialist
│   │   └── ...
│   ├── main.py               # CLI entry point & UI
│   └── orchestrator.py       # LangGraph orchestration & supervisor
├── pyproject.toml            # Package metadata & dependencies
└── README.md                 # You are here
```

## 🧠 How it Works

1. **User Input**: You type a request or a command (like `/add`).
2. **Supervisor**: A central "Source Control Agent" analyzes your request.
3. **Routing**: The supervisor routes the task to the best specialized agent (Coding, Math, etc.).
4. **Tool Use**: Agents can use tools or perform reasoning to fulfill the task.
5. **Memory**:
   - **Short-term**: Managed by LangGraph's `MemorySaver` (stays within a thread).
   - **Long-term**: Every interaction is vectorized and stored in **Pinecone**.
   - **Retrieval**: Before responding, agents query Pinecone for relevant past interactions to provide consistent, context-aware answers.

## ⌨️ Commands

| Command | Description |
| :--- | :--- |
| `/new` | Start a fresh conversational session |
| `/switch <id>` | Switch between active sessions |
| `/list` | List all active sessions and their IDs |
| `/add <path>` | Feed a file's content into the agent's context |
| `/help` | Display the welcome and help menu |
| `/exit` | Safely shutdown the application |

## 📝 License
MIT
