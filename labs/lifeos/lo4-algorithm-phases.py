#!/usr/bin/env python3
"""
LAB LO4: The Algorithm as a seven-phase state machine.

Every non-trivial task in LifeOS runs through the Algorithm: OBSERVE, THINK,
PLAN, BUILD, EXECUTE, VERIFY, LEARN, in that order. The order is the discipline.
You cannot verify before you build, and you cannot learn before you verify. In
this lab you implement the phase machine, prove it walks all seven phases in the
correct order, and prove it REFUSES to skip a phase. All synthetic.

Run: python3 modules/academy-content/labs/lifeos/lo4-algorithm-phases.py
"""
import sys

PHASES = ["OBSERVE", "THINK", "PLAN", "BUILD", "EXECUTE", "VERIFY", "LEARN"]

class Algorithm:
    def __init__(self):
        self.i = 0                 # index of the current phase
        self.visited = []

    def current(self):
        return PHASES[self.i]

    def advance(self, to):
        # only the very next phase in the sequence is allowed.
        expected = PHASES[self.i + 1] if self.i + 1 < len(PHASES) else None
        if to != expected:
            raise ValueError(f"cannot jump to {to}; next phase must be {expected}")
        self.i += 1
        self.visited.append(to)

# STEP 1: walk the whole loop in order, recording each phase entered.
a = Algorithm()
a.visited.append(a.current())
print("STEP 1: walking the phases in order")
for nxt in PHASES[1:]:
    a.advance(nxt)
    print(f"  entered {nxt}")

order_ok = (a.visited == PHASES)
print("")
print(f"STEP 2: visited all seven phases in order: {order_ok}")

# STEP 3: prove a skip is rejected. from OBSERVE you cannot leap to VERIFY.
b = Algorithm()
skip_rejected = False
try:
    b.advance("VERIFY")             # illegal: THINK is next, not VERIFY
except ValueError as e:
    skip_rejected = True
    print("")
    print(f"STEP 3: illegal skip rejected -> {e}")

ok = order_ok and skip_rejected
print("")
print(f"ALGORITHM ENFORCES PHASE ORDER (all seven, no skips): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Order is the discipline: observe, think, plan, build, execute, verify, learn. Next: skills.")
