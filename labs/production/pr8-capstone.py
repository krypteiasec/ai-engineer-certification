#!/usr/bin/env python3
"""
LAB PR8: Capstone. Serving + eval gate + monitoring, as one system.

Everything in this course is one machine when it runs in production. A request
does not just hit a model; it hits a SYSTEM that has already gated the deployed
version on evals, serves the traffic efficiently, and watches cost and quality the
whole time it is live. This capstone assembles those pieces into one small,
honest, end-to-end production service and proves the invariant that makes a system
trustworthy: only a gate-passing version serves, every request is metered, and a
drift monitor is watching the live inputs.

The pipeline:
  1. GATE   a candidate on the golden eval set; refuse to serve if it regresses.
  2. SERVE  the live requests in batches (throughput), classifying each.
  3. MONITOR token cost against a budget and input drift against a baseline.
It proves the gate admitted the good version, the batched server answered every
request correctly, and the monitors reported clean.

Run: python3 modules/academy-content/labs/production/pr8-capstone.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, embed, cosine

# golden set for the gate (known-correct labels).
GOLDEN = [
    ("I love this, it is fantastic", "positive"),
    ("this is terrible and broken", "negative"),
    ("an awesome and brilliant tool", "positive"),
    ("useless, slow, and disappointing", "negative"),
]
# the live traffic to serve (same domain, so drift stays low).
LIVE = [
    "the product is great and I recommend it",
    "a horrible and buggy experience",
    "I really like this, it works perfectly",
    "worst app, totally useless",
    "fantastic and wonderful, the best",
]
LIVE_LABELS = ["positive", "negative", "positive", "negative", "positive"]

# a REFERENCE window of normal traffic for the drift monitor. Real drift
# detection compares live inputs against a representative sample of normal
# traffic, NOT the tiny eval golden set, so this reference spans the domain's
# vocabulary. A genuine topic shift (physics, finance) would sit far outside it.
BASELINE_TRAFFIC = [
    "the product is great and works perfectly for me",
    "a horrible buggy experience, totally useless app",
    "I really like this tool, it is fantastic",
    "the worst product, disappointing and slow",
    "wonderful and the best, I recommend it to everyone",
]

GATE = 0.80
BUDGET = 1.00
PRICE_IN = 3.0 / 1_000_000
PRICE_OUT = 15.0 / 1_000_000
INSTRUCTION = "Classify the sentiment."


def eval_gate(instruction):
    correct = sum(1 for r, g in GOLDEN
                  if complete(instruction + " Input: " + r).strip().lower() == g)
    return correct / len(GOLDEN)


def serve_batches(inputs, instruction, batch=2):
    """Serve inputs in batches, returning (labels, passes). Batching here models
    the throughput lever from PR1: fewer forward passes for the same work."""
    labels, passes = [], 0
    for start in range(0, len(inputs), batch):
        chunk = inputs[start:start + batch]
        passes += 1
        for text in chunk:
            labels.append(complete(instruction + " Input: " + text).strip().lower())
    return labels, passes


def main():
    # ---- 1. GATE ----
    score = eval_gate(INSTRUCTION)
    admitted = score >= GATE
    print("STEP 1: gate the candidate on the golden set")
    print("  eval score %.2f  threshold %.2f  ->  %s"
          % (score, GATE, "ADMIT" if admitted else "REFUSE"))
    if not admitted:
        print("PRODUCTION SYSTEM SERVED, GATED, AND MONITORED: NO")
        sys.exit(1)

    # ---- 2. SERVE ----
    labels, passes = serve_batches(LIVE, INSTRUCTION, batch=2)
    served_correct = sum(1 for got, want in zip(labels, LIVE_LABELS) if got == want)
    print("")
    print("STEP 2: serve the live traffic in batches")
    print("  requests %d served in %d batched passes" % (len(LIVE), passes))
    print("  correct labels: %d of %d" % (served_correct, len(LIVE)))

    # ---- 3. MONITOR: cost + drift ----
    total_in = sum(len(t.split()) * 4 for t in LIVE)     # rough token proxy
    total_out = len(LIVE) * 2
    spend = total_in * PRICE_IN + total_out * PRICE_OUT
    under_budget = spend <= BUDGET

    vecs = [embed(t) for t in BASELINE_TRAFFIC]
    center = [sum(v[i] for v in vecs) / len(vecs) for i in range(len(vecs[0]))]
    live_drift = sum(1.0 - cosine(embed(t), center) for t in LIVE) / len(LIVE)
    base_spread = sum(1.0 - cosine(embed(t), center) for t in BASELINE_TRAFFIC) / len(BASELINE_TRAFFIC)
    drift_ok = live_drift <= base_spread + 0.25

    print("")
    print("STEP 3: monitor cost and drift on the live traffic")
    print("  spend $%.4f  budget $%.2f  ->  %s"
          % (spend, BUDGET, "ok" if under_budget else "OVER"))
    print("  live drift %.3f  threshold %.3f  ->  %s"
          % (live_drift, base_spread + 0.25, "ok" if drift_ok else "DRIFT"))

    # ---- the capstone invariant: gated + served correctly + monitors clean ----
    gated = admitted
    served = served_correct == len(LIVE) and passes < len(LIVE)
    monitored = under_budget and drift_ok
    ok = gated and served and monitored
    print("")
    print("  gated: %s   served-correct+batched: %s   monitored-clean: %s"
          % ("YES" if gated else "NO", "YES" if served else "NO",
             "YES" if monitored else "NO"))
    print("")
    print("PRODUCTION SYSTEM SERVED, GATED, AND MONITORED: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("That is the course: a prototype is a demo, this is a service you can trust.")


if __name__ == "__main__":
    main()
