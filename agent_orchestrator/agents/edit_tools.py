import os
import difflib
from langchain_core.tools import tool

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: Patch-based tools have been intentionally removed.
#
# The previous approach (apply_project_patch / multi_patch_project_file /
# patch_line_range) caused agent hallucination loops because the LLM would
# guess whitespace/indentation and the patch would silently fail, triggering
# retries forever.
#
# The new approach mirrors Claude Code:
#   1. plan_code_changes  – declare intent (non-destructive, shows plan to user)
#   2. write_project_file – rewrite the full file (100% reliable, no string matching)
# ──────────────────────────────────────────────────────────────────────────────

# Module-level store so the permission callback can access the last-seen plan
_last_plan: str = ""


@tool
def plan_code_changes(plan: str) -> str:
    """
    MUST be called before any write_project_file call.

    Declare the exact changes you intend to make:
      - Which files will be modified / created
      - What will change in each file (bullet-point summary)

    This tool is NON-DESTRUCTIVE. It only records your plan and presents it
    to the user for review. After calling this, wait for approval, then call
    write_project_file for each file.
    """
    global _last_plan
    _last_plan = plan
    return (
        "Plan approved by user. "
        "Now call write_project_file for EACH file listed in the plan — one file per tool call. "
        "Do NOT call plan_code_changes again."
    )


@tool
def write_project_file(file_path: str, content: str) -> str:
    """
    Writes the COMPLETE new content to a file.

    This is the ONLY write tool available. Always:
      1. Call read_project_file to get the current content
      2. Call plan_code_changes to declare your intent
      3. Call this tool with the fully updated file content

    NEVER write partial content or snippets — always provide the entire file.
    This approach eliminates string-matching failures entirely.
    """
    try:
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path) if os.path.dirname(abs_path) else ".", exist_ok=True)

        # Read existing content for the diff shown in the permission dialog
        old_content = ""
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                old_content = f.read()

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        lines_before = len(old_content.splitlines())
        lines_after = len(content.splitlines())
        action = "Created" if not old_content else "Updated"
        return f"{action} {file_path} ({lines_before} → {lines_after} lines)"
    except Exception as e:
        return f"Error writing file: {e}"


def compute_diff(file_path: str, new_content: str) -> str:
    """
    Returns a unified diff string between the current file content and new_content.
    Used by the permission UI in main.py.
    """
    abs_path = os.path.abspath(file_path)
    old_lines: list[str] = []
    if os.path.exists(abs_path):
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            old_lines = f.readlines()

    new_lines = [line if line.endswith("\n") else line + "\n" for line in new_content.splitlines()]

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "\n".join(list(diff))
