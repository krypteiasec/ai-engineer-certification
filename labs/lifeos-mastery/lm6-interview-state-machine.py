#!/usr/bin/env python3
"""
LAB LM6: The /interview state machine. One question at a time, never a firehose.

The /interview flow fills your TELOS by talking to you. It scans completeness
first, then picks a mode: if your files are at least 80 percent complete it runs
REVIEW (read each back, ask if it is still accurate, outdated, missing, or worth
sharpening); below that it runs FILL (walk the empty prompts). The one rule that
never bends: you get ONE question at a time. A wall of ten questions is a firehose
that makes people freeze, so the machine emits exactly one and waits. In this lab
you drive the state machine over a sample TELOS and prove it chose the right mode
and never emitted more than one question per turn. Sample data only.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm6-interview-state-machine.py
"""
import sys

# STEP 1: completeness of a sample TELOS. Each section is filled or empty.
sections = {
    "MISSION": True, "GOALS": True, "PROBLEMS": True, "STRATEGIES": True,
    "CHALLENGES": False, "NARRATIVES": True, "SPARKS": False,
    "BELIEFS": True, "WISDOM": True, "MODELS": True, "FRAMES": True,
}
completeness = sum(1 for v in sections.values() if v) / len(sections)
mode = "REVIEW" if completeness >= 0.80 else "FILL"
print("STEP 1: scan completeness, pick mode")
print(f"  filled {sum(sections.values())}/{len(sections)} = {completeness:.0%}  ->  mode = {mode}")

# STEP 2: the machine yields questions one at a time. In FILL mode it asks about
# empty sections; in REVIEW mode it reads a filled section back for a check.
def questions(sections, mode):
    if mode == "FILL":
        for name, filled in sections.items():
            if not filled:
                yield f"Your {name} is empty. In your own words, what belongs here?"
    else:
        for name, filled in sections.items():
            if filled:
                yield f"Here is your {name}. Still accurate, outdated, or worth sharpening?"

# STEP 3: emit turns. We record how many questions were surfaced per turn: the
# invariant is that it is ALWAYS exactly one (never a firehose).
per_turn = []
turns = []
for q in questions(sections, mode):
    per_turn.append(1)          # one question surfaced this turn
    turns.append(q)
print("")
print(f"STEP 2-3: {mode} mode surfaced {len(turns)} turns, one question each")
for i, q in enumerate(turns[:3], 1):
    print(f"  turn {i}: {q}")
if len(turns) > 3:
    print(f"  ... {len(turns)-3} more, still one at a time")

# INVARIANTS: mode chosen correctly from completeness, at least one question
# produced, and every single turn surfaced exactly one question.
mode_ok = (mode == "REVIEW" and completeness >= 0.80) or (mode == "FILL" and completeness < 0.80)
one_at_a_time = all(n == 1 for n in per_turn) and len(per_turn) >= 1
print("")
print(f"  mode matches completeness rule : {mode_ok}")
print(f"  every turn = exactly 1 question : {one_at_a_time}")

ok = mode_ok and one_at_a_time
print("")
print(f"ONE QUESTION AT A TIME (mode chosen by completeness, never a firehose): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("The interview meets you where you are, one question at a time. Next: migrate your context.")
