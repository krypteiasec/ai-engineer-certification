#!/usr/bin/env python3
"""
LAB CF3: Multi-agent orchestration, hooks, and escalation (Domain 1, 27%).

Hub-and-spoke: one coordinator delegates to subagents that have ISOLATED
context. Every fact a subagent needs must be passed explicitly. Hooks enforce
policy deterministically where prompts only enforce it probabilistically. And
some escalation triggers are reliable while others are exam traps. This lab
drills all three: it proves a subagent only answers correctly when context is
passed, routes subagent errors to the right recovery action, and separates
reliable escalation triggers from the unreliable distractors.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf3-orchestration.py
"""
import sys

# STEP 1: subagents have isolated context. A subagent cannot see the
# coordinator's history; it only knows what its own prompt contains.
CASE_FACTS = {"order_total": 89.99}


def subagent(prompt: dict):
    """A subagent sees ONLY its own prompt dict. It returns the order total if
    and only if the coordinator passed it explicitly."""
    return prompt.get("order_total", None)  # nothing inherited from the hub

no_context = subagent({"task": "summarize the order"})           # forgot to pass it
with_context = subagent({"task": "summarize", "order_total": CASE_FACTS["order_total"]})
print("STEP 1: isolated subagent context")
print(f"  no explicit context  -> {no_context} (subagent is blind)")
print(f"  explicit context     -> {with_context} (correct)")

# STEP 2: structured error routing. Each error category has a fixed retryable
# flag and coordinator action.
ERROR_TABLE = {
    "transient":  (True,  "retry with exponential backoff"),
    "validation": (True,  "fix input and retry"),
    "business":   (False, "explain to user, propose alternative"),
    "permission": (False, "escalate to human"),
}


def route_error(category: str):
    return ERROR_TABLE[category]

print("")
print("STEP 2: structured error routing")
routing_ok = True
for cat, (retryable, action) in ERROR_TABLE.items():
    got_retry, got_action = route_error(cat)
    ok_row = (got_retry == retryable and got_action == action)
    routing_ok = routing_ok and ok_row
    print(f"  {cat:<11} retryable={got_retry!s:<5} -> {got_action}")

# STEP 3: escalation triggers. Reliable vs trap.
RELIABLE = {"explicit_human_request", "policy_gap", "no_progress_after_attempts",
            "financial_over_threshold"}
TRAPS = {"negative_sentiment", "self_reported_confidence", "auto_category_classifier"}


def should_escalate(trigger: str) -> bool:
    """Only reliable triggers escalate. Traps must NOT drive escalation."""
    return trigger in RELIABLE

print("")
print("STEP 3: escalation triggers (reliable vs exam trap)")
escalation_ok = True
for t in sorted(RELIABLE):
    d = should_escalate(t)
    escalation_ok = escalation_ok and d
    print(f"  reliable  {t:<28} escalate={d}")
for t in sorted(TRAPS):
    d = should_escalate(t)
    escalation_ok = escalation_ok and (not d)
    print(f"  trap      {t:<28} escalate={d}")

# Invariants.
context_ok = (no_context is None and with_context == 89.99)
hook_beats_prompt = True  # hooks are deterministic(100%) vs prompts probabilistic
# demonstrate: a $700 refund is blocked by a hook every time, a prompt is not
HOOK_LIMIT = 500
refund = 700
hook_blocks = refund > HOOK_LIMIT  # PreToolUse hook redirects deterministically
print("")
print(f"STEP 4: hook enforcement on a ${refund} refund (limit ${HOOK_LIMIT})")
print(f"  PreToolUse hook redirects to escalation: {hook_blocks} (deterministic, every time)")

ok = context_ok and routing_ok and escalation_ok and hook_blocks
print("")
print(f"  explicit context is mandatory for subagents : {context_ok}")
print(f"  error categories route correctly            : {routing_ok}")
print(f"  reliable triggers escalate, traps do not     : {escalation_ok}")
print("")
print(f"ORCHESTRATION RULES HOLD (explicit context + error routing + escalation + hooks): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Pass context explicitly, enforce money with hooks, escalate on reliable triggers only. Next: Claude Code config.")
