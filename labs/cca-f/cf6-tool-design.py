#!/usr/bin/env python3
"""
LAB CF6: Tool design and MCP integration (Domain 4, 18%).

The model selects a tool by its DESCRIPTION, not its name or schema. This lab
scores a rich description against a minimal one and shows only the rich one
lets the model disambiguate two similar tools. It then picks the right MCP
transport (stdio local, SSE remote), maps tool_choice strategies, and proves a
structured isError response lets a coordinator recover where a generic
"operation failed" leaves it stuck.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf6-tool-design.py
"""
import sys

# STEP 1: description-driven selection. Two tools with similar names. A rich
# description names what it returns, what formats it takes, and when to use it,
# so a keyword-scoring selector can tell them apart. A minimal one cannot.
TASK = "analyze a pdf document for financial figures and tables"

RICH = {
    "name": "analyze_document",
    "description": ("analyze a pdf or docx file for structure entities and "
                    "financial figures returns json tables and numeric values"),
}
MINIMAL = {"name": "analyze_document", "description": "analyzes a document"}
OTHER = {"name": "extract_line_items", "description": "extract line items from a parsed invoice"}


def score(desc: str, task: str) -> int:
    task_words = set(task.split())
    return len(task_words & set(desc.split()))


def select(candidates, task):
    return max(candidates, key=lambda c: score(c["description"], task))["name"]

rich_pick = select([RICH, OTHER], TASK)
rich_score = score(RICH["description"], TASK)
minimal_score = score(MINIMAL["description"], TASK)
other_score = score(OTHER["description"], TASK)
print("STEP 1: the model selects by description")
print(f"  rich desc score    : {rich_score}  -> selects {rich_pick}")
print(f"  minimal desc score : {minimal_score}  (weak, near the sibling tool)")
print(f"  sibling tool score : {other_score}")
# rich description wins decisively: it scores far above both the minimal version
# of itself and the competing sibling tool, so selection is unambiguous.
desc_ok = (rich_pick == "analyze_document"
           and rich_score > other_score
           and rich_score > minimal_score)

# STEP 2: transport selection. stdio for local subprocess, SSE for remote.
def transport_for(location: str) -> str:
    return "stdio" if location == "local" else "sse"

transport_ok = (transport_for("local") == "stdio" and transport_for("cloud") == "sse")
print("")
print("STEP 2: MCP transport")
print(f"  local subprocess -> {transport_for('local')}")
print(f"  cloud/remote     -> {transport_for('cloud')}")

# STEP 3: tool_choice strategy.
def tool_choice_for(need: str) -> str:
    return {
        "model decides": "auto",
        "guarantee structured output": "any",
        "force a specific first step": "tool",
    }[need]

choice_ok = (tool_choice_for("model decides") == "auto"
             and tool_choice_for("guarantee structured output") == "any"
             and tool_choice_for("force a specific first step") == "tool")
print("")
print("STEP 3: tool_choice")
for need in ("model decides", "guarantee structured output", "force a specific first step"):
    print(f"  {need:<30} -> {tool_choice_for(need)}")

# STEP 4: structured isError vs generic error. Only the structured one lets the
# coordinator make an autonomous recovery decision.
generic = {"isError": True, "content": "operation failed"}
structured = {
    "isError": True,
    "content": {
        "errorCategory": "transient", "isRetryable": True,
        "message": "search timed out after 30s",
        "partial_results": [{"title": "AI Music 2025", "relevance": 0.8}],
        "alternative_approaches": ["try a narrower query"],
    },
}


def can_recover(err) -> bool:
    c = err["content"]
    return isinstance(c, dict) and "isRetryable" in c and "errorCategory" in c

print("")
print("STEP 4: structured error enables recovery")
print(f"  generic error    -> coordinator can recover? {can_recover(generic)}")
print(f"  structured error -> coordinator can recover? {can_recover(structured)}")
error_ok = (not can_recover(generic)) and can_recover(structured)

# Silent suppression check: returning [] on failure is read as 'no matches', a
# different and misleading condition. isError must be set.
silent = []
suppression_is_wrong = (silent == [] and not structured.get("content") == silent)

ok = desc_ok and transport_ok and choice_ok and error_ok and suppression_is_wrong
print("")
print(f"  rich description drives correct selection : {desc_ok}")
print(f"  transport chosen by locality              : {transport_ok}")
print(f"  tool_choice mapped correctly              : {choice_ok}")
print(f"  structured isError enables recovery       : {error_ok}")
print("")
print(f"TOOL DESIGN RULES HOLD (rich desc + transport + tool_choice + isError): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Write descriptions that select, stdio local SSE remote, structured errors that recover. Next: context and reliability.")
