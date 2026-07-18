#!/usr/bin/env python3
"""
CAPSTONE 3 (ch3): A tool-calling agent that solves a real multi-step task.

Capstone project two: an agent that does not just answer, it ACTS across several
tool calls to finish a workflow, and it stays inside a box while doing it. This is
the ReAct loop from the Agents course grown into a real task: a small trip-budget
assistant that, to answer "can I afford three nights in Tokyo and Paris on a $900
budget", has to look up two nightly rates, look up the nights, multiply and sum
across tool calls, and compare against the budget. No single call can do it; the
agent has to plan and chain.

It integrates the agent patterns (tool_route for routing, a tool registry, a
plan-act-observe loop) and adds the guardrail every real agent needs: a sensitive
"book the trip" action is on the registry but NOT on the allowlist, so when the
agent reaches for it the guard blocks it. The lab proves the agent completed the
multi-step calculation correctly AND that the ungranted action was refused.

Run: python3 modules/academy-content/labs/capstones/cap3-tool-agent.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route  # the routing move at the heart of ReAct
import re

# ── Tools the agent can call ─────────────────────────────────────────────────────
_RATES = {"tokyo": 120, "paris": 150, "london": 110}   # nightly hotel rate, USD


def hotel_rate(city):
    for name, rate in _RATES.items():
        if name in city.lower():
            return rate
    raise ValueError("no rate for %r" % city)


def calculator(expr):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*x])\s*(-?\d+(?:\.\d+)?)", expr)
    if not m:
        raise ValueError("cannot parse %r" % expr)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b}[op]
    return int(r) if r == int(r) else r


def book_trip(detail):   # SENSITIVE, irreversible-in-spirit: spends real money
    return "BOOKED: %s" % detail


REGISTRY = {"hotel": hotel_rate, "calculator": calculator, "book": book_trip}
ALLOWLIST = {"hotel", "calculator"}   # least privilege: booking is NOT granted


class TripAgent:
    def __init__(self):
        self.log = []

    def call(self, tool, arg):
        """Every call passes the guard: only allowlisted tools run."""
        if tool not in ALLOWLIST:
            self.log.append(("BLOCKED", tool))
            return {"ok": False, "blocked": True, "reason": "not on allowlist"}
        result = REGISTRY[tool](arg)
        self.log.append(("RAN", tool, result))
        return {"ok": True, "result": result}

    def solve(self, cities, nights, budget):
        print("TASK: afford %d nights in %s on a $%d budget?" %
              (nights, " and ".join(cities), budget))
        # PLAN: one hotel lookup per city, then multiply by nights, then sum.
        per_city = []
        for city in cities:
            # ROUTE: confirm the ask maps to the hotel tool, then ACT + OBSERVE.
            chosen = tool_route("look up the hotel rate for %s" % city, list(ALLOWLIST))
            rate = self.call(chosen if chosen else "hotel", city)["result"]
            cost = self.call("calculator", "%d * %d" % (rate, nights))["result"]
            print("  %s: $%d/night x %d nights = $%d" % (city, rate, nights, cost))
            per_city.append(cost)
        total = self.call("calculator", "%d + %d" % (per_city[0], per_city[1]))["result"]
        affordable = total <= budget
        print("  total across %d cities = $%d  ->  %s budget" %
              (len(cities), total, "within" if affordable else "over"))
        return total, affordable


def main():
    agent = TripAgent()
    total, affordable = agent.solve(["Tokyo", "Paris"], nights=3, budget=900)

    print("\nAgent then reaches for a sensitive action it was never granted:")
    blocked = agent.call("book", "Tokyo+Paris 3 nights")
    print("  book(...) -> %s" % blocked)

    # 3 hotel-cost multiplies + 1 sum, plus 2 hotel lookups = 6 successful runs.
    ran = sum(1 for e in agent.log if e[0] == "RAN")
    correct_total = (total == 810)     # (120*3) + (150*3) = 360 + 450
    right_verdict = affordable is True  # 810 <= 900
    blocked_ok = blocked.get("blocked") is True
    multi_step = ran >= 5              # genuinely chained several tool calls

    print("")
    print("  chained %d tool calls, total $%d correct, verdict correct: %s" % (
        ran, total, correct_total and right_verdict and multi_step))
    print("  blocked the ungranted booking action: %s" % ("YES" if blocked_ok else "NO"))

    ok = correct_total and right_verdict and multi_step and blocked_ok
    print("")
    print("AGENT COMPLETED THE MULTI-STEP TASK: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Plans, chains real tools, answers, and stays in its box. Capstone two.")


if __name__ == "__main__":
    main()
