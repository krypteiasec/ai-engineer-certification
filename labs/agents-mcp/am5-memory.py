#!/usr/bin/env python3
"""
LAB AM5: Memory and multi-step workflows.

A single model call has no memory. Say something on turn one and it is gone by
turn three. An agent that runs across turns needs WORKING MEMORY: a store that
carries facts forward so a later step can use what an earlier step learned. It
also needs a STEP CAP, because a loop with tools and no ceiling can run forever.
In this lab you have a three-turn conversation: turn one states a budget, later
turns do other work, and the final turn computes a percentage of that budget,
which only works if the agent remembered it. You prove memory carried the fact
across turns and the loop stayed inside its cap.

Run: python3 modules/academy-content/labs/agents-mcp/am5-memory.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import re

def calculator(expr):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*percent\s+of\s+(-?\d+(?:\.\d+)?)", expr, re.I)
    if m:
        return float(m.group(1)) / 100.0 * float(m.group(2))
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("cannot parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return r


class Agent:
    """A tiny stateful agent: it remembers facts and caps its own steps."""

    def __init__(self, max_steps=6):
        self.memory = {}          # WORKING MEMORY: facts learned this session
        self.steps = 0
        self.max_steps = max_steps

    def remember(self, key, value):
        self.memory[key] = value

    def act(self, turn):
        if self.steps >= self.max_steps:            # the runaway guard
            return "stopped: step cap reached"
        self.steps += 1
        # If the turn refers to a remembered fact, substitute it in before acting.
        if "my budget" in turn.lower() and "budget" in self.memory:
            turn = turn.lower().replace("my budget", str(self.memory["budget"]))
        tool = tool_route(turn, ["calculator", "search", "weather"])
        if tool == "calculator":
            return calculator(turn)
        return "noted"


agent = Agent()

# TURN 1: state a fact. The agent commits it to memory.
print("TURN 1: 'My budget is 500'")
agent.remember("budget", 500)
print(f"  memory now: {agent.memory}")

# TURN 2: unrelated work. Memory must survive it untouched.
print("")
print("TURN 2: 'Search the notes for the vendor list'")
r2 = agent.act("search the notes for the vendor list")
print(f"  result: {r2!r}   memory still: {agent.memory}")

# TURN 3: the payoff. Compute 20 percent of MY BUDGET, a fact from turn 1.
print("")
print("TURN 3: 'What is 20 percent of my budget?'")
r3 = agent.act("what is 20 percent of my budget")
print(f"  result: {r3!r}")

used_memory = (r3 == 100.0)             # 20% of 500 == 100, only if it recalled 500
within_cap = (agent.steps <= agent.max_steps)
print("")
print(f"final turn used the remembered budget : {used_memory}")
print(f"stayed within the step cap            : {within_cap} ({agent.steps}/{agent.max_steps})")

ok = used_memory and within_cap
print("")
print(f"AGENT USED MEMORY ACROSS TURNS: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Memory carries state; the cap prevents runaway. Next: the MCP protocol.")
