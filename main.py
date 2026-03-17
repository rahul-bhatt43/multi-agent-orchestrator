import os
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from orchestrator import run_orchestrator

# Load environment variables
load_dotenv()

def main():
    # Initialize LangFuse callback handler
    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

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
