#!/usr/bin/env python3
"""
LAB LO1: Current state to ideal state, the one idea LifeOS is built on.

LifeOS exists to move you from where you are now to where you want to be. It
articulates the ideal state as measurable criteria, then closes the gap. In this
lab you take a sample person's current numbers and their stated ideal, compute
the gap on each dimension, and prove that finishing the work drives the total gap
to zero. No personal data is used: the state below is a fictional sample the lab
creates itself.

Run: python3 modules/academy-content/labs/lifeos/lo1-current-to-ideal.py
"""
import sys

# STEP 1: a sample life state. current vs ideal on a few measurable dimensions.
# (All fictional. This is the shape of what TELOS captures, not anyone's real data.)
current = {"runway_months": 1, "shipped_products": 0, "weekly_deep_hours": 4}
ideal   = {"runway_months": 6, "shipped_products": 3, "weekly_deep_hours": 20}

def gaps(cur, goal):
    # the gap on each dimension is how far current sits below ideal (never below 0).
    return {k: max(0, goal[k] - cur[k]) for k in goal}

g = gaps(current, ideal)
print("STEP 1-2: gap on each dimension (ideal minus current)")
for k in sorted(g):
    print(f"  {k:20} current={current[k]:>3}  ideal={ideal[k]:>3}  gap={g[k]:>3}")

total_gap_start = sum(g.values())
print("")
print(f"  total gap now: {total_gap_start}")

# STEP 3: do the work. moving current up to ideal closes every gap to zero.
after = dict(ideal)                       # the work lands current on the ideal
g_after = gaps(after, ideal)
total_gap_end = sum(g_after.values())
print("")
print(f"STEP 3: after reaching ideal on every dimension, total gap: {total_gap_end}")

# INVARIANT: there was a real gap to close, and closing it drives the total to 0.
ok = (total_gap_start > 0) and (total_gap_end == 0)
print("")
print(f"CURRENT STATE CONVERGES TO IDEAL (gap {total_gap_start} -> {total_gap_end}): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("That is the whole engine of LifeOS. Next: install it and run the interview.")
