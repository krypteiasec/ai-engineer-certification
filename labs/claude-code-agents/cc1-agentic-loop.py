#!/usr/bin/env python3
"""
LAB CC1: The agentic loop. Gather context, act, observe, verify, repeat, stop.

Claude Code is an agent harness, and the Claude Agent SDK is that same harness as
a library: you call query(prompt, options) and it runs a loop that gathers
context, takes an action with a tool, observes the result, and repeats until the
goal is met. The engine of every agent is that loop, and the one thing a loop
MUST have is a stop condition, a verify step that asks "is the goal satisfied
yet". Without it an agent either quits too early or runs forever. In this lab you
build the minimal loop by hand: a goal that needs two facts, a verify predicate
that checks whether both are known, and a decide step that either stops or picks
the next action. You prove the loop reaches a verified-complete state and stops
inside its step budget.

Reference (real Claude Agent SDK, not run here, offline mock below):
    from claude_agent_sdk import query, ClaudeAgentOptions
    async for message in query(prompt="report the weather in Paris and Tokyo",
                               options=ClaudeAgentOptions(allowed_tools=["Bash"])):
        print(message)

Run: python3 modules/academy-content/labs/claude-code-agents/cc1-agentic-loop.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route

# ── The world the agent can act on: a weather tool. ─────────────────────────────
_WEATHER = {"paris": 22, "tokyo": 30, "london": 15}

def weather(city):
    for name, temp in _WEATHER.items():
        if name in city.lower():
            return temp
    raise ValueError("no weather for %r" % city)

TOOLS = ["weather", "search"]

# The GOAL and its verify predicate. The goal is "know the temperature in both
# Paris and Tokyo". `state` is the agent's running memory of what it has learned.
NEEDED = ["paris", "tokyo"]
MAX_STEPS = 6

def goal_met(state):
    """VERIFY: the stop condition. True only when every needed fact is known."""
    return all(city in state for city in NEEDED)

def decide(state):
    """DECIDE: return the next city to look up, or None if nothing is left."""
    for city in NEEDED:
        if city not in state:
            return city
    return None

# ── The loop itself: gather -> decide -> act -> observe -> verify -> repeat. ─────
state = {}          # what the agent has learned so far (its context)
steps = 0
print(f"GOAL: know the temperature in {NEEDED}")
print("")

while steps < MAX_STEPS:
    if goal_met(state):                 # VERIFY first: stop the moment we are done
        print(f"STEP {steps}: verify -> goal met, stopping")
        break
    steps += 1
    city = decide(state)                # DECIDE the next action
    tool = tool_route(f"look up the weather in {city}", TOOLS)  # pick the tool
    observation = weather(city)         # ACT + OBSERVE
    state[city] = observation           # update context
    print(f"STEP {steps}: decide -> {city!r}")
    print(f"        act    -> {tool}({city!r})")
    print(f"        observe-> {observation}")
    print(f"        state  -> {dict(state)}")

print("")
answer = {c: state.get(c) for c in NEEDED}
print(f"final state : {answer}")
print(f"steps taken : {steps} (cap {MAX_STEPS})")

verified_complete = goal_met(state)
within_budget = steps <= MAX_STEPS
correct = answer == {"paris": 22, "tokyo": 30}

print("")
print(f"reached a verified-complete state : {verified_complete}")
print(f"stopped inside the step budget    : {within_budget}")
print(f"the learned facts are correct     : {correct}")

ok = verified_complete and within_budget and correct
print("")
print(f"AGENTIC LOOP TERMINATED IN A VERIFIED-COMPLETE STATE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Gather, act, observe, verify, stop. Next: defining the tools the loop calls.")
