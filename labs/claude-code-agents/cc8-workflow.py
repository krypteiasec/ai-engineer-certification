#!/usr/bin/env python3
"""
LAB CC8: The complete agentic workflow. Everything at once.

This is the capstone. You assemble every piece of the course into one agent that
does real, multi-step work and stays inside its box. It PLANS the goal into
steps, calls TOOLS through a PreToolUse GUARD, DELEGATES one focused piece to a
SUBAGENT with its own context, VERIFIES the result against the truth, and REPORTS
a final answer. The goal: read a sales note, extract the deal size, double it for
a projection, and get an independent sentiment read on the note, all while a
guardrail blocks an action the agent was never granted. In this lab you run the
whole pipeline end to end and prove it produced the correct verified result while
the guard stopped the ungranted action, which is exactly what a production agent
must do: work AND stay safe.

Reference (real Claude Agent SDK ties loop + tools + subagents + hooks together
under query(); the executed lab builds it offline on the mock):
    from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

Run: python3 modules/academy-content/labs/claude-code-agents/cc8-workflow.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete
import re

NOTE = "Deal with Acme closed. My name is Jordan and the number is 40. Great client, excellent to work with."

# ── Tools + the PreToolUse guard (least privilege). ─────────────────────────────
def extract(_):     return complete("Extract JSON: %s" % NOTE)   # pull facts as JSON
def calculator(a):  return a * 2                                  # double it
REGISTRY = {"extract": extract, "calculator": calculator}
ALLOWLIST = {"extract", "calculator"}     # 'send_email' is deliberately not granted
guard_log = []

def guarded_call(tool, arg):
    """PreToolUse guard: run only allowlisted tools; log every decision."""
    if tool not in ALLOWLIST:
        guard_log.append(("BLOCKED", tool))
        return {"ok": False, "blocked": True}
    guard_log.append(("RAN", tool))
    return {"ok": True, "result": REGISTRY[tool](arg)}

# ── A subagent: an independent sentiment read on the note (isolated context). ───
def sentiment_subagent(text):
    return complete("system: you classify sentiment\nuser: sentiment: %s" % text)

# ── PLAN -> ACT (guarded) -> DELEGATE -> VERIFY -> REPORT. ───────────────────────
print("GOAL: extract the deal size, project double, and read the note's sentiment")
print("")

# Step 1: extract facts through the guard.
import json
ex = guarded_call("extract", NOTE)
facts = json.loads(ex["result"])
deal = facts["number"]
print(f"step 1 extract  -> deal size = {deal}")

# Step 2: double it through the guard.
proj = guarded_call("calculator", deal)
projection = proj["result"]
print(f"step 2 project  -> {deal} x 2 = {projection}")

# Step 3: delegate the sentiment read to a subagent.
mood = sentiment_subagent(NOTE)
print(f"step 3 delegate -> sentiment = {mood!r}")

# Step 4: the agent tries an action it was never granted; the guard blocks it.
blocked = guarded_call("send_email", "the projection is %s" % projection)
print(f"step 4 ungranted send_email -> {blocked}")

# Step 5: verify against the truth and report.
report = {"deal": deal, "projection": projection, "sentiment": mood}
print("")
print(f"REPORT: {report}")

correct = report == {"deal": 40, "projection": 80, "sentiment": "positive"}
guard_blocked = blocked.get("blocked") is True
ran_two_tools = sum(1 for e in guard_log if e[0] == "RAN") == 2

print("")
print(f"produced the correct verified report : {correct}")
print(f"guard blocked the ungranted action   : {guard_blocked}")
print(f"exactly the two granted tools ran     : {ran_two_tools}")

ok = correct and guard_blocked and ran_two_tools
print("")
print(f"COMPLETE AGENTIC WORKFLOW PRODUCED THE VERIFIED RESULT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Plan, act under a guard, delegate, verify, report. You have finished the course.")
