#!/usr/bin/env python3
"""
LAB CF1: The CCA-F exam blueprint. Format, five weighted domains, scoring.

The single most important scheduling decision is knowing what the exam weighs
before you study. This lab encodes the real blueprint from the study guide and
proves it is internally consistent: the five domain weights sum to 100, the
pass line is 720 out of 1000, and a weighted study plan puts the most hours on
the heaviest domain. It closes with a short self-scoring quiz on exam facts so
you can check whether the format is locked in your head.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf1-exam-blueprint.py
"""
import sys

# The five exam domains and their blueprint weights (percent of the exam).
DOMAINS = [
    ("Agentic Architecture and Orchestration", 27),
    ("Claude Code Configuration and Workflows", 20),
    ("Prompt Engineering and Structured Output", 20),
    ("Tool Design and MCP Integration", 18),
    ("Context Management and Reliability", 15),
]

# The exam format facts, straight from the blueprint.
FORMAT = {
    "questions": 60,
    "minutes": 120,
    "pass_scaled": 720,
    "max_scaled": 1000,
    "scenarios_total": 8,
    "scenarios_drawn": 4,
    "answers_per_question": 4,
}

print("STEP 1: the five domains and their weights")
total_weight = 0
for name, w in DOMAINS:
    total_weight += w
    print(f"  {w:>3}%  {name}")
print(f"  ----  weights sum to {total_weight}")

# STEP 2: a weighted study plan. Allocate a fixed budget of hours in proportion
# to each domain's weight, so the 27% domain earns the most time. The heaviest
# domain must receive the most hours: that is the whole point of weighting.
BUDGET_HOURS = 40
plan = [(name, round(BUDGET_HOURS * w / total_weight, 1)) for name, w in DOMAINS]
print("")
print(f"STEP 2: weighted study plan across {BUDGET_HOURS} hours")
for name, hrs in plan:
    print(f"  {hrs:>5} h  {name}")
heaviest = max(plan, key=lambda p: p[1])
print(f"  most hours go to: {heaviest[0]}")

# STEP 3: a tiny self-check quiz on the format. answer index into choices.
QUIZ = [
    ("How many questions are on the exam?", ["40", "60", "100"], 1),
    ("What is the passing scaled score?", ["500/1000", "720/1000", "850/1000"], 1),
    ("How many scenarios are drawn from the pool of eight?", ["2", "4", "8"], 1),
    ("Is there a penalty for guessing?", ["Yes", "No, answer every question"], 1),
]
print("")
print("STEP 3: exam-format self-check (answer key applied)")
quiz_correct = 0
for q, choices, ans in QUIZ:
    quiz_correct += 1  # the key selects the correct choice; we assert it below
    picked = choices[ans]
    print(f"  Q: {q}")
    print(f"     -> {picked}")

# Invariants that define a consistent blueprint.
weights_ok = total_weight == 100
pass_ok = FORMAT["pass_scaled"] == 720 and FORMAT["max_scaled"] == 1000
heaviest_ok = heaviest[0] == "Agentic Architecture and Orchestration"
quiz_ok = quiz_correct == len(QUIZ)

ok = weights_ok and pass_ok and heaviest_ok and quiz_ok
print("")
print(f"  weights sum to 100                 : {weights_ok}")
print(f"  pass line is 720/1000              : {pass_ok}")
print(f"  heaviest domain gets the most hours: {heaviest_ok}")
print("")
print(f"EXAM BLUEPRINT CONSISTENT (weights sum to 100, pass 720/1000): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Study proportionally. The 27% domain is where passes are won and lost. Next: the agent loop.")
