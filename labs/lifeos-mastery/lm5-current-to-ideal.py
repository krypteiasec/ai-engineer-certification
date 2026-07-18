#!/usr/bin/env python3
"""
LAB LM5: Current state to ideal state. The engine under everything.

The whole of LifeOS is one loop: understand where you are, understand where you
want to be, and hill-climb from one to the other. Everything else is mechanics.
TELOS records both states across five life dimensions: HEALTH, MONEY, FREEDOM,
RELATIONSHIPS, CREATIVE. The gap between current and ideal is the number the
system exists to shrink. In this lab you score that gap for a fictional person,
then run a hill-climb that does the work one step at a time, and prove the total
gap falls to zero. All numbers are invented for the lesson.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm5-current-to-ideal.py
"""
import sys

DIMS = ["HEALTH", "MONEY", "FREEDOM", "RELATIONSHIPS", "CREATIVE"]

# STEP 1: current and ideal, scored 0..100 per dimension. Fictional sample.
current = {"HEALTH": 55, "MONEY": 40, "FREEDOM": 30, "RELATIONSHIPS": 60, "CREATIVE": 45}
ideal   = {"HEALTH": 90, "MONEY": 85, "FREEDOM": 80, "RELATIONSHIPS": 85, "CREATIVE": 90}

def total_gap(state):
    return sum(ideal[d] - state[d] for d in DIMS)

print("STEP 1: score the gap per dimension")
for d in DIMS:
    print(f"  {d:14} current={current[d]:3}  ideal={ideal[d]:3}  gap={ideal[d]-current[d]:3}")
start_gap = total_gap(current)
print(f"  total gap = {start_gap}")

# STEP 2: hill-climb. Each step, do the work on the dimension with the LARGEST
# remaining gap (the highest-leverage move) and close part of it. Repeat.
state = dict(current)
gaps_over_time = [total_gap(state)]
STEP = 5
steps = 0
while total_gap(state) > 0:
    worst = max(DIMS, key=lambda d: ideal[d] - state[d])
    state[worst] = min(ideal[worst], state[worst] + STEP)
    gaps_over_time.append(total_gap(state))
    steps += 1
    if steps > 1000:      # safety, should never trigger
        break

print("")
print("STEP 2: hill-climb, always work the largest gap")
print(f"  gap trajectory (every 10th step): {gaps_over_time[::10] + [gaps_over_time[-1]]}")
print(f"  steps taken = {steps}")

# STEP 3: invariants. The gap moved monotonically DOWN (never got worse), it
# reached exactly zero, and the final state matches the ideal on every dimension.
monotone_down = all(gaps_over_time[i+1] <= gaps_over_time[i] for i in range(len(gaps_over_time)-1))
reached_zero = total_gap(state) == 0
matches_ideal = all(state[d] == ideal[d] for d in DIMS)
print("")
print(f"STEP 3: gap only ever decreased       : {monotone_down}")
print(f"        total gap reached zero        : {reached_zero}")
print(f"        final state equals ideal      : {matches_ideal}")

ok = monotone_down and reached_zero and matches_ideal and start_gap > 0
print("")
print(f"GAP CLOSES (hill-climb drove the total gap to zero across 5 dimensions): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Current to ideal, one leveraged step at a time. Next: the interview that fills TELOS.")
