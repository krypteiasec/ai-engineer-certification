#!/usr/bin/env python3
"""
LAB LO8: Compose the whole thing into a daily operating system.

This is the capstone. Everything you learned wires together: a TELOS goal sets
the target, the Algorithm decomposes it into ISCs (testable criteria), work runs
and each ISC is verified, and memory routes what was learned into the right tier
(WORK, KNOWLEDGE, LEARNING) so tomorrow starts smarter. In this lab you run one
full synthetic day and prove the loop closes: goal advanced, every ISC verified,
and each learning routed to a real tier. Fully fictional sample data.

Run: python3 modules/academy-content/labs/lifeos/lo8-daily-os.py
"""
import sys

# STEP 1: a TELOS goal decomposed into Ideal State Criteria (each a yes/no probe).
goal = "G0: Ship the MVP"
iscs = [
    {"id": "ISC-1", "text": "API endpoint returns 200", "passed": False},
    {"id": "ISC-2", "text": "landing page renders",     "passed": False},
    {"id": "ISC-3", "text": "signup writes a user row",  "passed": False},
]

# STEP 2: execute and verify. each criterion flips to passed only with evidence.
evidence = {"ISC-1": "curl -> 200", "ISC-2": "screenshot ok", "ISC-3": "SELECT found row"}
print(f"STEP 1-2: pursue {goal}")
for c in iscs:
    c["passed"] = c["id"] in evidence     # verified by a (simulated) probe
    print(f"  {c['id']} {c['text']:28} -> {'PASS' if c['passed'] else 'pending'}  ({evidence.get(c['id'],'')})")

all_passed = all(c["passed"] for c in iscs)
progress = f"{sum(c['passed'] for c in iscs)}/{len(iscs)}"

# STEP 3: memory routing. each learning goes to exactly one tier.
TIERS = {"WORK", "KNOWLEDGE", "LEARNING"}
learnings = [
    ("MVP shipped, ISCs all green", "WORK"),
    ("vector search beats keyword here", "KNOWLEDGE"),
    ("reproduce before fixing saved an hour", "LEARNING"),
]
routed = {tier: [] for tier in TIERS}
for text, tier in learnings:
    routed[tier].append(text)              # router places each learning in one tier
print("")
print("STEP 3: route each learning to its memory tier")
for tier in sorted(routed):
    print(f"  {tier:10} <- {routed[tier]}")

# INVARIANT: the day's loop closed. every ISC verified, so the goal advanced, and
# every learning landed in exactly one valid tier (nothing lost, nothing misfiled).
routing_ok = (
    all(t in TIERS for _, t in learnings)
    and sum(len(v) for v in routed.values()) == len(learnings)
)
ok = all_passed and routing_ok
print("")
print(f"DAILY OS LOOP CLOSED (goal {progress} verified, learnings routed): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Goal in, verified work out, memory compounding. That is LifeOS as a daily operating system.")
