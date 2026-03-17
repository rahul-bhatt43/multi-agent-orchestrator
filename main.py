import os
from dotenv import load_dotenv
try:
    from langfuse.callback import CallbackHandler
except ImportError:
    from langfuse import Langfuse
    # We'll need to adapt this if CallbackHandler is missing, 
    # but for now let's see if we can at least import something.
    CallbackHandler = None 
from orchestrator import run_orchestrator

# Load environment variables
load_dotenv()

def main():
    # Initialize LangFuse callback handler if available
    langfuse_handler = None
    if CallbackHandler is not None:
        langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
    else:
        print("Warning: LangFuse CallbackHandler not found. Tracing disabled.")

    print("--- Multi-Agent Orchestration System ---")
    
    # Example queries
    queries = [
        "What is 256 * 12 + 10?",
        "Write a Python function to check if a number is prime.",
        "What is the capital of France?",
        # "What do we know about agent-orchestration in the knowledge base?" # Potential knowledge base query
    ]

    for query in queries:
        print(f"\nUser: {query}")
        try:
            response = run_orchestrator(query, langfuse_handler=langfuse_handler)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
