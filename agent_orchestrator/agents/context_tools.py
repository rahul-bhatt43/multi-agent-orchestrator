import os
import fnmatch
from typing import List, Dict, Optional
from langchain_core.tools import tool

# Directories to ignore by default
DEFAULT_IGNORE_PATTERNS = [
    'node_modules', '.venv', '.git', '.next', '__pycache__', 
    'dist', 'build', '.pytest_cache', '.ipynb_checkpoints',
    '*.pyc', '*.pyo', '*.pyd', '.env', '.env.*', '.ds_store'
]

def should_ignore(path: str, root_dir: str, extra_ignores: List[str] = []) -> bool:
    """Checks if a path should be ignored based on common patterns."""
    rel_path = os.path.relpath(path, root_dir)
    patterns = DEFAULT_IGNORE_PATTERNS + extra_ignores
    
    # Check each part of the path
    parts = rel_path.split(os.sep)
    for part in parts:
        for pattern in patterns:
            if fnmatch.fnmatch(part.lower(), pattern.lower()):
                return True
    return False

@tool
def list_project_structure(directory: str = "."):
    """
    Returns a recursively generated file tree of the project, 
    skipping ignored directories like node_modules and .venv.
    Use this to understand the project layout.
    """
    root_dir = os.path.abspath(directory)
    tree = []
    
    for root, dirs, files in os.walk(root_dir):
        # Filter directories in-place to prevent os.walk from descending into ignored ones
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), root_dir)]
        
        rel_root = os.path.relpath(root, root_dir)
        if rel_root == ".":
            rel_root = ""
            
        for file in files:
            file_path = os.path.join(root, file)
            if not should_ignore(file_path, root_dir):
                tree.append(os.path.join(rel_root, file))
                
    return "\n".join(sorted(tree))

@tool
def read_project_file(file_path: str):
    """
    Reads the content of a specific file in the project.
    Use this to examine the code of a specific module.
    """
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' not found."
            
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Basic safety: if file is huge, warn
            if len(content) > 50000:
                return content[:50000] + "\n\n... [File truncated for length] ..."
            return content
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def search_in_project(query: str, extension_filter: Optional[str] = None):
    """
    Searches for a string pattern across all non-ignored files in the project.
    Similar to 'grep'. Use this to find where a function or variable is used.
    """
    root_dir = os.path.abspath(".")
    results = []
    
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), root_dir)]
        
        for file in files:
            if extension_filter and not file.endswith(extension_filter):
                continue
                
            file_path = os.path.join(root, file)
            if should_ignore(file_path, root_dir):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if query.lower() in line.lower():
                            rel_path = os.path.relpath(file_path, root_dir)
                            results.append(f"{rel_path}:{i}: {line.strip()}")
                            if len(results) > 50: # Cap results
                                return "\n".join(results) + "\n... [Too many results, please refine search] ..."
            except:
                continue
                
@tool
def read_project_file_with_lines(file_path: str) -> str:
    """Reads a project file and returns its content with line numbers (e.g., '1: content'). Useful for precise patching."""
    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' not found."
            
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        output = []
        for i, line in enumerate(lines, 1):
            output.append(f"{i}: {line}")
            
        return "".join(output)
    except Exception as e:
        return f"Error reading file with lines: {e}"
