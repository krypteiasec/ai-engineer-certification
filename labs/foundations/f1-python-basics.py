#!/usr/bin/env python3
"""
LAB F1: Your first Python. Variables, types, and printing.

Every program starts by holding values in variables and looking at them. That is
the whole job of this lab: make some variables, see their types, print them
clearly, and prove a couple of facts about them. No imports, no magic.

Run: python3 modules/academy-content/labs/foundations/f1-python-basics.py
"""
import sys

# STEP 1: variables hold values. Python figures out the type for you.
name = "Aria"          # a string of text
chapters = 8          # an integer
pi = 3.14159          # a floating point number
learning = True       # a boolean, True or False
print("STEP 1: variables and their types")
for label, value in [("name", name), ("chapters", chapters), ("pi", pi), ("learning", learning)]:
    print(f"  {label} = {value!r}   (type {type(value).__name__})")

# STEP 2: f-strings put values into text, and arithmetic works as you expect.
minutes_each = 20
total = chapters * minutes_each
print("")
print("STEP 2: arithmetic and f-strings")
print(f"  {chapters} chapters x {minutes_each} min = {total} minutes total")

# STEP 3: the invariant. A program that cannot check its own facts is not much use.
ok = (2 + 2 == 4) and (total == 160) and (type(name).__name__ == "str")
print("")
print(f"STEP 3: basic facts hold (2+2==4, total==160, name is str): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("You wrote and ran real Python. Next: lists and loops, how Python handles many values.")
