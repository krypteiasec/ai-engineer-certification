#!/usr/bin/env python3
"""
LAB CC3: The ReAct pattern in practice. Reason, act, observe, chain, stop.

One tool call rarely finishes a real task, because real tasks have steps whose
later actions depend on earlier results. ReAct (Reason + Act) is how an agent
handles that: in a loop it REASONS about what it needs next, ACTS by calling a
tool, OBSERVES the result into a scratchpad, and feeds that observation into the
next reasoning step. It stops when the goal is answered or a step cap is hit. The
key move is CHAINING: step two's input is step one's observation. In this lab the
agent answers a question no single tool can, "how many days until launch, times
two". It searches for the day count, observes 7, then chains that 7 into the
calculator to get 14, proving the reasoning chain reached the right answer.

Reference (real Claude Agent SDK runs this loop for you inside query();
the executed lab builds it by hand on the offline mock):
    async for message in query(prompt="days until launch times two",
                               options=ClaudeAgentOptions(allowed_tools=["Bash"])):
        ...

Run: python3 modules/academy-content/labs/claude-code-agents/cc3-react.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route

# ── Tools: a knowledge search and a calculator. ─────────────────────────────────
_FACTS = {"days until launch": 7, "team size": 5}

def search(query):
    for key, val in _FACTS.items():
        if key in query.lower():
            return val
    raise ValueError("nothing found for %r" % query)

def calculator(a, b):
    return a * b

TOOLS = ["search", "calculator"]
GOAL = "the number of days until launch, times two"
TRUTH = 14
MAX_STEPS = 4

# The agent's plan as ordered sub-goals. A real agent derives these on the fly;
# here they are explicit so the reason -> act -> observe -> chain flow is legible.
plan = [
    "search how many days until launch",   # -> search, observes 7
    "calculate {obs} times 2",             # -> calculator, filled from the observation
]

scratchpad = []   # OBSERVE store: what the agent has learned this run
answer = None
steps = 0

print(f"GOAL: {GOAL}  (truth = {TRUTH})")
print("")
for subgoal in plan:
    if steps >= MAX_STEPS:
        break
    steps += 1
    # CHAIN: fill this step's thought from the previous observation.
    thought = subgoal.format(obs=scratchpad[-1]["observation"]) if scratchpad else subgoal
    tool = tool_route(thought, TOOLS)                    # REASON: pick the tool
    if tool == "search":
        observation = search(thought)                    # ACT + OBSERVE
    else:
        observation = calculator(scratchpad[-1]["observation"], 2)
    scratchpad.append({"thought": thought, "tool": tool, "observation": observation})
    print(f"STEP {steps}: reason -> {thought!r}")
    print(f"        act    -> {tool}()")
    print(f"        observe-> {observation}")
    answer = observation

print("")
print(f"scratchpad : {[s['observation'] for s in scratchpad]}")
print(f"final answer: {answer}")
print(f"steps taken : {steps} (cap {MAX_STEPS})")

chained = len(scratchpad) == 2 and scratchpad[0]["observation"] == 7
correct = answer == TRUTH
within_budget = steps <= MAX_STEPS

print("")
print(f"chained observation 7 into step two : {chained}")
print(f"final answer equals the truth 14     : {correct}")
print(f"stopped inside the step budget       : {within_budget}")

ok = chained and correct and within_budget
print("")
print(f"REACT REASONING CHAIN REACHED THE CORRECT ANSWER: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Reason, act, observe, chain, stop. Next: hand focused work to subagents.")
