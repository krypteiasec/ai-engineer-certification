#!/usr/bin/env python3
"""
LAB LM4: The goal validator. A goal is measurable; a wish is not.

TELOS draws a hard line. "Ship the MVP by 2026-Q2 with 100 daily active users"
is a goal: it names a deliverable, a date, and a number you can check. "Be better
at security" is a wish: nothing to measure, no date, no done. Only goals carry an
Ideal State Criterion, the binary test that answers "did this happen, yes or no".
In this lab you run a validator over a sample list and prove every real goal
passes and every wish fails, for the right reason. Sample text only.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm4-goal-validator.py
"""
import sys, re

# STEP 1: a candidate is a goal only if it is measurable. The checkable target is
# a NUMBER (how many, by when); a DATE (year, quarter, timeframe) is a bonus
# signal; and a "be/get/feel better" phrasing marks a WISH with nothing to test.
# The rule: a goal has a countable target and is not a vague wish.
DATE = re.compile(r"\b(20\d{2}|Q[1-4]|this (year|month|week)|every \w+ (week|day|month))\b", re.I)
NUMBER = re.compile(r"\b\d+\b")
WISH = re.compile(r"\b(be|get|become|feel)\s+(better|good|great|more)\b", re.I)

def validate(text):
    has_date = bool(DATE.search(text))
    has_number = bool(NUMBER.search(text))
    is_wish = bool(WISH.search(text))
    measurable = has_number and not is_wish     # countable target, not a wish
    return measurable, {"date": has_date, "number": has_number, "wish": is_wish}

# STEP 2: a labeled sample set. `expect` is the ground truth for the invariant.
samples = [
    ("Ship the MVP of Project X by 2026-Q2, 100 daily active users", True),
    ("Publish 24 essays this year, one every other week", True),
    ("Bank a 6 month cash runway of 32000 dollars by 2026", True),
    ("Be better at security", False),
    ("Get more disciplined", False),
    ("Feel more confident someday", False),
]
print("STEP 1-2: validate each candidate goal")
results = []
for text, expect in samples:
    measurable, why = validate(text)
    results.append(measurable == expect)
    verdict = "GOAL " if measurable else "wish "
    print(f"  [{verdict}] {text[:52]:52}  date={int(why['date'])} num={int(why['number'])} wish={int(why['wish'])}")

# STEP 3: invariant. Every candidate was classified as the ground truth says, so
# real goals (date + number, not a wish) pass and wishes fail.
all_correct = all(results)
goals_pass = validate("Ship X by 2026-Q2, 100 DAU")[0] is True
wish_fails = validate("be better at X")[0] is False
print("")
print(f"STEP 3: all {len(samples)} classified correctly : {all_correct}")
print(f"        'Ship X by 2026-Q2, 100 DAU' passes : {goals_pass}")
print(f"        'be better at X' fails              : {wish_fails}")

ok = all_correct and goals_pass and wish_fails
print("")
print(f"GOAL VALIDATOR (measurable goals pass, wishes fail): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A goal you can test is a goal your DA can pursue. Next: the rest of TELOS.")
