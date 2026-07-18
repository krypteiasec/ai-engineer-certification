#!/usr/bin/env python3
"""
LAB AM2: The ReAct loop. Reason, act, observe, repeat.

One tool call is not enough for a real task, because real tasks have steps. The
ReAct pattern is the engine of every agent: in a loop, the agent REASONS about
what it needs next, ACTS by calling a tool, OBSERVES the result, and repeats
until it can answer. It keeps a scratchpad of what it has learned so far, and it
STOPS when the goal is met or a step cap is hit. In this lab the agent answers a
two-step question ("the temperature in Paris, plus 10") that no single tool call
can solve: it looks up the weather, observes the number, then calls the
calculator, and only then answers, proving the loop reached the right result.

Run: python3 modules/academy-content/labs/agents-mcp/am2-react-loop.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import re

# ── Tools: a weather dict-lookup and a calculator. ──────────────────────────────
_WEATHER = {"paris": ("sunny", 22), "tokyo": ("clear", 30), "london": ("rainy", 15)}

def weather(city):
    key = city.strip().lower()
    for name, (cond, temp) in _WEATHER.items():
        if name in key:
            return temp
    raise ValueError("no weather for %r" % city)

def calculator(expr):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("cannot parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r

def dispatch(tool, arg):
    return {"weather": weather, "calculator": calculator}[tool](arg)

TOOLS = ["weather", "calculator", "search"]
GOAL = "the temperature in Paris plus 10"
TRUTH = 32
MAX_STEPS = 5

# The agent's PLAN: the ordered sub-goals it will reason through. A real agent
# derives these as it goes; here they are explicit so the loop is easy to read.
plan = [
    "look up the weather temperature in Paris",   # -> weather tool
    "compute {temp} + 10",                          # -> calculator, filled from observation
]

scratchpad = []   # OBSERVE store: what the agent has learned this run
answer = None
steps = 0

print(f"GOAL: {GOAL}  (truth = {TRUTH})")
print("")
for i, subgoal in enumerate(plan):
    if steps >= MAX_STEPS:            # the stop condition: never run away
        break
    steps += 1
    # Fill the sub-goal from what we already observed (chain step 2 onto step 1).
    thought = subgoal.format(temp=scratchpad[-1]["observation"]) if scratchpad else subgoal
    tool = tool_route(thought, TOOLS)                       # REASON: pick a tool
    observation = dispatch(tool, thought)                  # ACT + OBSERVE
    scratchpad.append({"thought": thought, "tool": tool, "observation": observation})
    print(f"STEP {steps}: reason -> {thought!r}")
    print(f"         act    -> {tool}()")
    print(f"         observe-> {observation!r}")
    answer = observation   # the last observation is the running answer

print("")
print(f"final answer : {answer!r}")
print(f"reached in   : {steps} steps (cap {MAX_STEPS})")

ok = (answer == TRUTH) and (steps <= MAX_STEPS)
print("")
print(f"REACT LOOP REACHED THE CORRECT ANSWER: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Reason, act, observe, repeat, stop. Next: designing the tools themselves.")
