#!/usr/bin/env python3
"""
LAB LO5: Validate a skill's frontmatter and route by trigger.

A LifeOS skill is a folder with a SKILL.md at its front door. The frontmatter
declares the skill's name and a description whose USE WHEN triggers let the system
route a request to it automatically. In this lab you parse a sample skill's
frontmatter, validate it (name present, description present and non-trivial), and
match a user request against its triggers to prove routing works. Both skills
below are invented for the lesson.

Run: python3 modules/academy-content/labs/lifeos/lo5-skill-frontmatter.py
"""
import sys

# STEP 1: two sample SKILL.md front-matter blocks. One valid, one broken.
GOOD = {
    "name": "Council",
    "description": "Multi-agent debate. USE WHEN council, debate, weigh options, get different views.",
}
BAD = {
    "name": "",                       # missing name
    "description": "todo",            # too thin to route on
}

def validate(fm):
    problems = []
    if not fm.get("name"):
        problems.append("missing name")
    desc = fm.get("description", "")
    if len(desc) < 20:
        problems.append("description too thin")
    if "USE WHEN" not in desc:
        problems.append("no USE WHEN triggers")
    return problems

good_problems = validate(GOOD)
bad_problems = validate(BAD)
print("STEP 1: validate frontmatter")
print(f"  GOOD skill problems: {good_problems}")
print(f"  BAD  skill problems: {bad_problems}")

# STEP 2: routing. pull the triggers after 'USE WHEN' and match a request.
def triggers(desc):
    if "USE WHEN" not in desc:
        return []
    tail = desc.split("USE WHEN", 1)[1]
    return [t.strip(" .").lower() for t in tail.replace(".", ",").split(",") if t.strip(" .")]

request = "can you convene a debate on this decision"
trigs = triggers(GOOD["description"])
routed = any(t in request for t in trigs)
print("")
print(f"STEP 2: request routes to Council via a trigger: {routed}  (matched on 'debate')")

# STEP 3: invariants. valid skill passes clean, broken skill is caught, and the
# valid skill's triggers actually route a matching request.
ok = (good_problems == []) and (len(bad_problems) >= 2) and routed
print("")
print(f"SKILL VALIDATED AND ROUTED: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("A good SKILL.md validates clean and routes itself. Next: hooks that fire on events.")
