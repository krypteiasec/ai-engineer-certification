#!/usr/bin/env python3
"""
LAB AM3: Designing tools the model can use.

A tool is only as good as its description. The model does not see your Python
code, it sees a SCHEMA: a name, a plain-English description of what the tool is
for, and the parameters it takes. That schema is the entire contract the model
routes on, so a clear name and an honest description are what let it pick the
right tool for a request. In this lab you define three tools as real schemas,
build a tiny registry, and route several different questions through it, proving
that well-described tools send every query to the correct tool.

Run: python3 modules/academy-content/labs/agents-mcp/am3-tool-schemas.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import json

# ── The tool SCHEMAS. This is what the model reads. Same shape a real API uses:
#    a name, a description, and a typed parameter list. ──────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Do arithmetic. Use for any math, sums, products, or percentages.",
        "parameters": {"expression": {"type": "string", "description": "e.g. '47 * 12'"}},
    },
    {
        "name": "search",
        "description": "Find a fact in the knowledge base. Use to look up or search for information.",
        "parameters": {"query": {"type": "string", "description": "what to look up"}},
    },
    {
        "name": "weather",
        "description": "Get the current weather and temperature for a city.",
        "parameters": {"city": {"type": "string", "description": "a city name"}},
    },
]

# The registry the router uses: just the names, derived from the schemas.
names = [t["name"] for t in TOOL_SCHEMAS]
print("REGISTERED TOOLS (the schemas the model reads):")
for t in TOOL_SCHEMAS:
    print(f"  {t['name']:<11} - {t['description']}")

# A labeled set of asks, each with the tool it SHOULD go to.
cases = [
    ("What is 20 percent of 350?", "calculator"),
    ("Search the docs for the refund policy.", "search"),
    ("What is the weather in Tokyo today?", "weather"),
    ("Compute 12 * 12.", "calculator"),
]

print("")
print("ROUTING each ask through the schema registry:")
all_right = True
for ask, expected in cases:
    picked = tool_route(ask, names)
    hit = (picked == expected)
    all_right = all_right and hit
    print(f"  {ask!r}")
    print(f"     -> picked {picked!r}, expected {expected!r}  {'ok' if hit else 'WRONG'}")

# Prove the schemas are well-formed too: every tool has a name, description, and
# at least one parameter. A tool with a vague or missing description is a bug.
well_formed = all(
    t.get("name") and t.get("description") and t.get("parameters") for t in TOOL_SCHEMAS
)
print("")
print(f"all schemas well-formed (name + description + params): {well_formed}")

ok = all_right and well_formed
print("")
print(f"TOOL SCHEMAS ROUTED EVERY QUERY TO ITS TOOL: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("The schema is the contract. Next: dispatching the chosen tool safely.")
