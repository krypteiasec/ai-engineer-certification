#!/usr/bin/env python3
"""
LAB AM1: What an agent is. A model in a loop that can use tools.

A plain model call is a closed box: text in, text out, and nothing else happens.
It cannot look anything up, cannot run a calculation, cannot touch the world. An
AGENT is that same model given TOOLS and the permission to call them. The
smallest possible agent is a model plus one tool: read the ask, decide a tool is
needed, call it, and use the result. In this lab you ask for a real calculation
two ways, once from the bare model (which can only echo words back) and once from
a one-tool agent (which calls a calculator and returns the real number), and you
prove the tool is what turns a talker into a doer.

Run: python3 modules/academy-content/labs/agents-mcp/am1-what-is-an-agent.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, tool_route
import re

# ── The tool. A tool is just a plain function the agent is allowed to call. This
#    one does arithmetic the model itself cannot do reliably. ──────────────────
def calculator(expr):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("calculator could not parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r

QUESTION = "Compute 47 * 12"
TRUTH = 564

# 1. BARE MODEL: no tools. Asked to compute, it has no calculator, so it falls
#    back to echoing the words of the prompt. It cannot produce the number.
bare = complete(QUESTION)
print("STEP 1: the bare model, no tools")
print(f"  question : {QUESTION!r}")
print(f"  output   : {bare!r}")
bare_right = str(TRUTH) in bare

# 2. ONE-TOOL AGENT: read the ask, ROUTE to a tool, CALL it, use the result.
#    That decide-then-act is the seed of every agent.
TOOLS = ["calculator", "search", "weather"]
print("")
print("STEP 2: the one-tool agent")
chosen = tool_route(QUESTION, TOOLS)          # reason: which tool does this need?
print(f"  chose tool : {chosen!r}")
result = calculator(QUESTION) if chosen == "calculator" else None  # act: call it
print(f"  tool result: {result!r}")
agent_right = (result == TRUTH)

# 3. The tool is the whole difference. Same model, but now it can act.
print("")
print(f"STEP 3: bare model got the number   : {bare_right}")
print(f"        one-tool agent got it       : {agent_right}")

ok = agent_right and not bare_right
print("")
print(f"AGENT WITH ONE TOOL SOLVED WHAT THE BARE MODEL COULD NOT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("An agent is a model that can act. Next: put that decision in a loop (ReAct).")
