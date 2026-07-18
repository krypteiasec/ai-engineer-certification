#!/usr/bin/env python3
"""
LAB LM8: Maturity self-assessment and the Algorithm's phase order.

Two ideas close the course. First, the PAI Maturity Model: systems climb from
Chatbots (CH1-3) to Agents (AG1-3) to Assistants (AS1-3), and the target is AS3,
the Trusted Companion, a DA you route your whole life through. You can score your
own level from the capabilities you actually have. Second, the Algorithm: the way
LifeOS runs hard work is seven phases in a fixed order, OBSERVE, THINK, PLAN,
BUILD, EXECUTE, VERIFY, LEARN, each producing a hard-to-vary explanation. Run them
out of order and the work breaks. In this lab you score a sample capability set to
a maturity level and prove the seven phases execute in the one correct order.
Sample data only.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm8-maturity-and-algorithm.py
"""
import sys

# STEP 1: score maturity from capabilities. More trusted, autonomous, life-wide
# capabilities lift you from Chatbot toward AS3, the Trusted Companion.
caps = {
    "answers_questions": True,     # CH: a chatbot
    "uses_tools": True,            # AG: an agent that acts
    "runs_loops": True,            # AG
    "knows_your_telos": True,      # AS: it knows YOU
    "acts_across_your_life": True, # AS
    "trusted_with_autonomy": True, # AS3: the trusted companion
}
def maturity(c):
    if c["trusted_with_autonomy"] and c["acts_across_your_life"] and c["knows_your_telos"]:
        return "AS3"
    if c["knows_your_telos"] or c["acts_across_your_life"]:
        return "AS1"
    if c["uses_tools"] or c["runs_loops"]:
        return "AG2"
    return "CH1"
level = maturity(caps)
print("STEP 1: score maturity from capabilities")
for k, v in caps.items():
    print(f"  {k:22} {v}")
print(f"  -> level = {level}  (target is AS3, the Trusted Companion)")

# STEP 2: the Algorithm. Seven phases, one correct order. We execute them and log
# the order they ran, then check it against the canonical sequence.
CANONICAL = ["OBSERVE", "THINK", "PLAN", "BUILD", "EXECUTE", "VERIFY", "LEARN"]
ran = []
def phase(name):
    ran.append(name)
for p in CANONICAL:          # a correct run walks them in order
    phase(p)
print("")
print("STEP 2: run the Algorithm phases")
print(f"  ran: {' -> '.join(ran)}")

# STEP 3: invariants. Maturity scored to AS3 from the capability set, the seven
# phases ran, and they ran in exactly the canonical order (OBSERVE first, LEARN
# last, nothing skipped or swapped).
level_ok = level == "AS3"
order_ok = ran == CANONICAL
count_ok = len(ran) == 7
observe_first = ran[0] == "OBSERVE"
learn_last = ran[-1] == "LEARN"
print("")
print(f"STEP 3: maturity scored to AS3        : {level_ok}")
print(f"        7 phases ran in exact order   : {order_ok and count_ok}")
print(f"        OBSERVE first, LEARN last     : {observe_first and learn_last}")

ok = level_ok and order_ok and count_ok and observe_first and learn_last
print("")
print(f"MATURITY SCORED AND ALGORITHM ORDERED (level from capabilities, 7 phases in order): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("You have set up LifeOS end to end and can run it as a daily operating system. That is the course.")
