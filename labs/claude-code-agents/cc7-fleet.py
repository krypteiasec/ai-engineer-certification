#!/usr/bin/env python3
"""
LAB CC7: Orchestrating a fleet. Decompose, fan out in bounded waves, verify.

One agent is a worker. An orchestrator is the agent that runs a team of them. The
pattern is four beats: DECOMPOSE a goal into independent workstreams, FAN OUT the
workstreams to worker agents, VERIFY each worker's output, and SYNTHESIZE the
verified results into one answer. The catch is concurrency: firing a hundred
agents at once trips rate limits and melts throughput, so a fleet runs in BOUNDED
WAVES, never more than a fixed cap of workers at a time. In this lab an
orchestrator splits a ten-item job, runs it in waves of at most three, verifies
every worker output before accepting it, and synthesizes a final tally, proving
all ten completed, every wave respected the cap, and the synthesis is correct.

Reference (real Claude Agent SDK: each worker is a subagent invocation via the
Agent tool; the executed lab simulates the fleet offline and deterministically):
    agents={"worker": AgentDefinition(description=..., prompt=..., tools=[...])}

Run: python3 modules/academy-content/labs/claude-code-agents/cc7-fleet.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# The job: classify ten reviews. Each is an independent workstream.
JOB = [
    "I love it", "this is terrible", "great product", "worst ever, broken",
    "excellent and fantastic", "awful and useless", "the best, amazing",
    "boring and slow", "wonderful, recommend", "disappointing, poor",
]
CAP = 3   # bounded waves: never more than this many workers at once

def worker(item):
    """One worker agent: classify a single item. Returns its result."""
    return complete(f"sentiment: {item}")

def verify(item, result):
    """The orchestrator checks each worker output before accepting it."""
    return result in ("positive", "negative", "neutral")

def waves(items, cap):
    """DECOMPOSE into bounded waves of at most `cap` items each."""
    return [items[i:i + cap] for i in range(0, len(items), cap)]

# ── Orchestrate: run wave by wave, verify each output, synthesize. ──────────────
print(f"job has {len(JOB)} items, wave cap = {CAP}")
print("")
verified = []
wave_sizes = []
for w, batch in enumerate(waves(JOB, CAP), 1):
    wave_sizes.append(len(batch))
    print(f"wave {w}: {len(batch)} workers")
    for item in batch:                      # FAN OUT across the wave
        result = worker(item)               # a worker agent runs
        if verify(item, result):            # VERIFY before accepting
            verified.append(result)

# SYNTHESIZE: one tally from all verified worker outputs.
summary = {
    "positive": verified.count("positive"),
    "negative": verified.count("negative"),
    "neutral": verified.count("neutral"),
}
print("")
print(f"wave sizes  : {wave_sizes}")
print(f"synthesis   : {summary}")

all_done = len(verified) == len(JOB)
cap_respected = all(s <= CAP for s in wave_sizes)
correct = summary == {"positive": 5, "negative": 5, "neutral": 0}

print("")
print(f"all items completed and verified : {all_done}")
print(f"every wave respected the cap      : {cap_respected}")
print(f"synthesis tally is correct        : {correct}")

ok = all_done and cap_respected and correct
print("")
print(f"FLEET COMPLETED ALL TASKS IN BOUNDED WAVES: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Decompose, fan out, verify, synthesize. Next: the complete workflow.")
