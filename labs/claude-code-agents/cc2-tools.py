#!/usr/bin/env python3
"""
LAB CC2: Defining tools. The function-calling contract.

An agent is only as capable as the tools you give it, and a tool is a contract
with four parts: a name, a description, an input schema, and the function that
runs. The model reads the name and description to decide WHEN to call the tool,
the schema says what arguments are valid, and your harness runs the function. The
model chooses; the harness executes. Two rules earn their keep. Write the
description to say WHEN to call it ("Call this to add two numbers"), not just
what it is, because a prescriptive description raises the model's should-call
accuracy. And validate every call against the schema BEFORE running it, so a
malformed call is rejected at the boundary instead of crashing your function. In
this lab you build a real tool registry, gate every call through its schema, and
prove a valid call runs while a call missing a required argument is refused
without ever executing.

Reference (real Claude Agent SDK tool shape; executed lab is the offline mock):
    tools=[{ "name": "add", "description": "Call this to add two numbers.",
             "input_schema": {"type": "object",
                 "properties": {"a": {"type":"number"}, "b": {"type":"number"}},
                 "required": ["a", "b"]}}]

Run: python3 modules/academy-content/labs/claude-code-agents/cc2-tools.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route

# ── The tool registry: name -> {description, schema, fn}. ────────────────────────
def _add(a, b):      return a + b
def _weather(city):  return {"paris": 22, "tokyo": 30, "london": 15}[city.lower()]

REGISTRY = {
    "calculator": {
        "description": "Call this to add two numbers a and b.",
        "schema": {"required": ["a", "b"], "types": {"a": (int, float), "b": (int, float)}},
        "fn": lambda args: _add(args["a"], args["b"]),
    },
    "weather": {
        "description": "Call this to get the temperature for a city.",
        "schema": {"required": ["city"], "types": {"city": (str,)}},
        "fn": lambda args: _weather(args["city"]),
    },
}

def validate(tool, args):
    """The schema GATE. Returns (ok, reason). No function runs until this passes."""
    spec = REGISTRY[tool]["schema"]
    for key in spec["required"]:
        if key not in args:
            return False, f"missing required arg {key!r}"
    for key, val in args.items():
        if key in spec["types"] and not isinstance(val, spec["types"][key]):
            return False, f"arg {key!r} has wrong type"
    return True, "ok"

def call_tool(tool, args):
    """Every tool call passes the schema gate before the function executes."""
    ok, reason = validate(tool, args)
    if not ok:
        return {"ok": False, "ran": False, "reason": reason}
    result = REGISTRY[tool]["fn"](args)
    return {"ok": True, "ran": True, "result": result}

# ── 1. The description drives selection: the model picks the right tool. ─────────
chosen = tool_route("please calculate 2 plus 2", list(REGISTRY.keys()))
print(f"routing 'calculate 2 plus 2' -> {chosen!r}")

# ── 2. A well-formed call runs. ─────────────────────────────────────────────────
good = call_tool("calculator", {"a": 2, "b": 2})
print(f"calculator(a=2, b=2)        -> {good}")

# ── 3. A call missing a required arg is REJECTED at the schema gate, not run. ────
bad = call_tool("calculator", {"a": 2})
print(f"calculator(a=2)  [no b]     -> {bad}")

print("")
selected_right = chosen == "calculator"
valid_ran = good["ran"] and good["result"] == 4
invalid_blocked = (bad["ran"] is False) and (bad["ok"] is False)

print(f"description selected the right tool   : {selected_right}")
print(f"valid call ran and returned 4         : {valid_ran}")
print(f"invalid call blocked before executing : {invalid_blocked}")

ok = selected_right and valid_ran and invalid_blocked
print("")
print(f"TOOL CONTRACT ENFORCED (schema gates execution): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Name, description, schema, function. Next: the ReAct pattern that chains them.")
