#!/usr/bin/env python3
"""
LAB AM4: Tool calling and function dispatch.

Routing picks a tool by name. DISPATCH is what actually runs it: map the chosen
name to the real Python function, call it with the model's arguments, and feed
the result back into the loop. The part beginners skip is failure. A model can
name a tool that does not exist, or hand a tool an argument it cannot use, and a
naive dispatcher crashes the whole agent. A real dispatcher catches both and
returns a safe error the agent can reason about. In this lab you dispatch a good
call, an unknown tool, and a tool that throws, and prove the agent survives all
three.

Run: python3 modules/academy-content/labs/agents-mcp/am4-tool-dispatch.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
import re

def calculator(expression):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expression)
    if not m:
        raise ValueError("calculator could not parse %r" % expression)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r

REGISTRY = {"calculator": calculator}   # name -> real function

def dispatch(name, args):
    """Run a tool by name, ALWAYS returning a result dict, never raising. This
    is the boundary that keeps one bad tool call from killing the agent."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": "unknown tool: %r" % name}      # bad name
    try:
        return {"ok": True, "result": fn(**args)}                     # success
    except Exception as e:                                             # tool threw
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}

# 1. A GOOD call: known tool, valid args. Returns a result.
good = dispatch("calculator", {"expression": "6 * 7"})
print("STEP 1: good call     ->", good)

# 2. An UNKNOWN tool: the model hallucinated a name we do not have.
unknown = dispatch("teleporter", {"x": 1})
print("STEP 2: unknown tool  ->", unknown)

# 3. A tool that THROWS: known tool, but the argument makes no sense.
threw = dispatch("calculator", {"expression": "banana"})
print("STEP 3: tool threw    ->", threw)

# The three invariants of a safe dispatcher:
good_ok = good["ok"] and good["result"] == 42                    # good call worked
unknown_handled = (not unknown["ok"]) and "unknown tool" in unknown["error"]
threw_handled = (not threw["ok"]) and "error" in threw            # failure captured
survived = True   # if we reached here, nothing crashed the process

print("")
print(f"good call returned 42            : {good_ok}")
print(f"unknown tool handled, no crash  : {unknown_handled}")
print(f"tool exception caught, no crash : {threw_handled}")

ok = good_ok and unknown_handled and threw_handled and survived
print("")
print(f"DISPATCHER EXECUTED TOOLS AND HANDLED FAILURES SAFELY: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A tool call can fail. The dispatcher makes that survivable. Next: memory.")
