#!/usr/bin/env python3
"""
LAB AM8: The full agent, with guardrails.

This is the capstone. You assemble everything into one agent that answers a
multi-step question by PLANNING (break the goal into tool steps), CALLING tools
in order, and combining the results. But autonomy is the risk: an agent with
tools and no limits is a liability. So the agent runs under GUARDRAILS. Least
privilege: it may only call tools on an allowlist. Human-in-the-loop: any
sensitive action is gated and refused without explicit approval. In this lab the
agent computes the combined temperature of two cities across three tool calls,
and, when it tries a destructive action it was never granted, the guard blocks
it, proving the agent both works and stays inside its box.

Run: python3 modules/academy-content/labs/agents-mcp/am8-guarded-agent.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import re

# ── Tools ──────────────────────────────────────────────────────────────────────
_WEATHER = {"paris": 22, "tokyo": 30, "london": 15}

def weather(city):
    for name, temp in _WEATHER.items():
        if name in city.lower():
            return temp
    raise ValueError("no weather for %r" % city)

def calculator(expr):
    m = re.search(r"sum of\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", expr, re.I)
    if m:
        return int(float(m.group(1)) + float(m.group(2)))
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("cannot parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r

def send_email(body):
    return "email sent: %s" % body   # a SENSITIVE, irreversible-in-spirit action

REGISTRY = {"weather": weather, "calculator": calculator, "email": send_email}

# ── Guardrails ─────────────────────────────────────────────────────────────────
ALLOWLIST = {"weather", "calculator"}   # least privilege: email is NOT granted
SENSITIVE = {"email"}                    # requires explicit human approval


class GuardedAgent:
    def __init__(self):
        self.log = []

    def call(self, tool, arg, approved=False):
        """Every tool call passes through the guard first."""
        if tool not in ALLOWLIST:
            self.log.append(("BLOCKED", tool, "not on allowlist"))
            return {"ok": False, "blocked": True, "reason": "not on allowlist"}
        if tool in SENSITIVE and not approved:
            self.log.append(("GATED", tool, "needs approval"))
            return {"ok": False, "blocked": True, "reason": "needs approval"}
        result = REGISTRY[tool](arg)
        self.log.append(("RAN", tool, result))
        return {"ok": True, "result": result}

    def plan(self, goal):
        """Decompose a multi-step goal into ordered tool steps."""
        cities = re.findall(r"\b(Paris|Tokyo|London)\b", goal)
        steps = [("weather", c) for c in cities]              # one lookup per city
        steps.append(("calculator", None))                    # then combine
        return steps

    def run(self, goal):
        print(f"GOAL: {goal}")
        steps = self.plan(goal)
        print(f"PLAN: {[s[0] for s in steps]}")
        temps = []
        for tool, arg in steps:
            if tool == "weather":
                out = self.call("weather", arg)
                print(f"  weather({arg}) -> {out}")
                temps.append(out["result"])
            elif tool == "calculator":
                expr = "sum of %d and %d" % (temps[0], temps[1])
                out = self.call("calculator", expr)
                print(f"  calculator({expr!r}) -> {out}")
                answer = out["result"]
        return answer


agent = GuardedAgent()
answer = agent.run("What is the sum of the temperatures in Paris and Tokyo?")
print(f"\nANSWER: {answer}")

# Now the agent tries something it was never granted: emailing the result out.
print("\nAgent attempts a sensitive action it lacks permission for:")
blocked = agent.call("email", "the answer is %s" % answer)   # not approved, not allowlisted
print(f"  email(...) -> {blocked}")

planned_and_answered = (answer == 52)                        # 22 + 30, across 3 tool calls
tool_calls = sum(1 for entry in agent.log if entry[0] == "RAN")
blocked_ok = blocked.get("blocked") is True                  # the guard stopped it
ran_enough = tool_calls == 3                                 # two weather + one calculator

print("")
print(f"planned, called 3 tools, answered 52 : {planned_and_answered and ran_enough}")
print(f"blocked the ungranted email action   : {blocked_ok}")

ok = planned_and_answered and ran_enough and blocked_ok
print("")
print(f"GUARDED AGENT PLANNED, CALLED TOOLS, AND ANSWERED: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A real agent acts AND stays in its box. You have finished Agents, Tools and MCP.")
