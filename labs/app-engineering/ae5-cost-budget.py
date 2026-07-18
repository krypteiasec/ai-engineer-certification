#!/usr/bin/env python3
"""
LAB AE5: Cost control. Token accounting and a budget guard.

You pay per token, and an LLM feature with no cost ceiling is a bill waiting to
happen. The engineering answer is a budget guard: estimate the cost of a call
BEFORE you make it, track cumulative spend, and refuse a call that would blow the
budget instead of discovering it on the invoice. In this lab you price calls
against a per-model rate card, run several cheap calls that fit the budget, then
watch the guard block a large call that would exceed it, before a single token is
spent on it.

Run: python3 modules/academy-content/labs/app-engineering/ae5-cost-budget.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


# Dollars per 1,000 tokens. A small model is cheap, a large one is not.
PRICES = {
    "academy-mini":  {"in": 0.0005, "out": 0.0015},
    "academy-large": {"in": 0.003,  "out": 0.015},
}


def count_tokens(text):
    return len(text.split())


def estimate_cost(model, in_tokens, out_tokens):
    p = PRICES[model]
    return (in_tokens / 1000.0) * p["in"] + (out_tokens / 1000.0) * p["out"]


class BudgetExceeded(Exception):
    """Raised BEFORE spending when a call would push cumulative cost over the
    budget. Failing loud and early beats an unbounded bill."""


class CostGuard:
    def __init__(self, budget_usd):
        self.budget = budget_usd
        self.spent = 0.0
        self.log = []

    def call(self, model, prompt, max_out_tokens=32):
        in_tok = count_tokens(prompt)
        # Guard on the WORST case (full max_out_tokens) before making the call.
        projected = estimate_cost(model, in_tok, max_out_tokens)
        if self.spent + projected > self.budget:
            raise BudgetExceeded(
                "blocked: spent $%.4f + projected $%.4f > budget $%.4f"
                % (self.spent, projected, self.budget))
        reply = complete(prompt)
        actual = estimate_cost(model, in_tok, count_tokens(reply))
        self.spent += actual
        self.log.append({"model": model, "cost": round(actual, 6)})
        return reply


guard = CostGuard(budget_usd=0.01)

print("STEP 1: several small calls that fit the budget")
prompts = [
    "Classify the sentiment. Input: I love this product",
    "Classify the sentiment. Input: this update is terrible and broken",
    "Classify the sentiment. Input: the interface is nice and fast",
]
served = 0
for p in prompts:
    reply = guard.call("academy-mini", p, max_out_tokens=8)
    served += 1
    print("  ok  -> %-9r  running spend $%.6f" % (reply, guard.spent))

print("")
print("STEP 2: a large call that would exceed the budget")
blocked = False
try:
    guard.call("academy-large",
               "Extract JSON from this very long report " * 20,
               max_out_tokens=5000)
except BudgetExceeded as e:
    blocked = True
    print("  ", e)

print("")
print("STEP 3: result")
print("  calls served under budget :", served)
print("  total spent               : $%.6f" % guard.spent)
print("  over-budget call blocked  :", blocked)

ok = (served == 3) and blocked and (guard.spent > 0.0) and (guard.spent <= guard.budget)
print("")
print("COST GUARD BLOCKED THE OVER-BUDGET CALL: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Estimate before you spend, track the running total, refuse the overage. Next: route across providers.")
