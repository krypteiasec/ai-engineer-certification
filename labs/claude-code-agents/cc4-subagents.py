#!/usr/bin/env python3
"""
LAB CC4: Subagents and delegation. Isolated context, focused work, report back.

When a job splits into independent pieces, one agent doing all of it in a single
context gets slow and confused: every piece pollutes the context of every other.
The fix is subagents. The main agent delegates a focused subtask to a subagent
that has its OWN context window, its own instructions, and its own tool set. The
subagent works in isolation and reports only its result back. Two payoffs:
context isolation (a subagent sees only its slice, so nothing bleeds across), and
fan-out (independent pieces run as separate agents). In the real Claude Agent SDK
you define one with AgentDefinition(description=, prompt=, tools=, model=) and the
main agent invokes it through the Agent tool. In this lab an orchestrator
delegates three reviews, one per subagent, and you prove each subagent's context
held ONLY its own review while all three labels came back correct.

Reference (real Claude Agent SDK; the executed lab simulates it offline):
    agents={"reviewer": AgentDefinition(
        description="Classify the sentiment of one review.",
        prompt="You classify sentiment.", tools=["Read"])}

Run: python3 modules/academy-content/labs/claude-code-agents/cc4-subagents.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# An AgentDefinition: the subagent's persona and the ONE tool it is granted.
SUBAGENT = {
    "description": "Classify the sentiment of a single review.",
    "prompt": "You are a sentiment classifier. Read one review and label it.",
    "tools": ["classify"],
}

# The independent work items. Each goes to its own subagent, in isolation.
REVIEWS = [
    "I love this product, it is excellent",
    "this is terrible and broken, worst ever",
    "the app is fantastic and I recommend it",
]
TRUTH = ["positive", "negative", "positive"]


def spawn_subagent(defn, task):
    """Run ONE subagent in an isolated context. It receives only `task`, builds
    its own local transcript, and returns its result plus that transcript. Note
    it never sees the other tasks: that is context isolation, enforced by only
    passing this one slice in."""
    transcript = [
        f"system: {defn['prompt']}",
        f"user: sentiment of this review: {task}",
    ]
    label = complete("\n".join(transcript))     # the subagent's own tool call
    transcript.append(f"assistant: {label}")
    return {"result": label, "context": transcript}


# ── Orchestrator: delegate each item to its own subagent, collect the results. ──
print(f"orchestrator has {len(REVIEWS)} independent items, one subagent each")
print("")
results = []
contexts = []
for i, review in enumerate(REVIEWS, 1):
    out = spawn_subagent(SUBAGENT, review)
    results.append(out["result"])
    contexts.append(out["context"])
    print(f"subagent {i}: task={review[:32]!r}... -> {out['result']!r}")

print("")
print(f"collected labels: {results}")

# Isolation check: subagent i's context must contain its OWN review and NONE of
# the others. If any other review leaked in, isolation is broken.
isolation_ok = True
for i, ctx in enumerate(contexts):
    blob = "\n".join(ctx)
    if REVIEWS[i] not in blob:
        isolation_ok = False
    for j, other in enumerate(REVIEWS):
        if j != i and other in blob:
            isolation_ok = False

all_reported = len(results) == len(REVIEWS)
all_correct = results == TRUTH

print("")
print(f"each subagent saw only its own slice : {isolation_ok}")
print(f"all subagents reported back          : {all_reported}")
print(f"all labels correct                   : {all_correct}")

ok = isolation_ok and all_reported and all_correct
print("")
print(f"SUBAGENTS RAN IN ISOLATED CONTEXTS AND ALL REPORTED BACK: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Delegate, isolate, report. Next: plug in external tools over MCP.")
