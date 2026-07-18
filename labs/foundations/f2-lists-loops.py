#!/usr/bin/env python3
"""
LAB F2: Lists and loops. Working with many values at once.

Machine learning is lists of numbers all the way down. Here you build lists, walk
them with loops, transform them with comprehensions, and prove the results. This
is the muscle you use in every later lab.

Run: python3 modules/academy-content/labs/foundations/f2-lists-loops.py
"""
import sys

# STEP 1: a list holds an ordered sequence of values.
nums = [4, 8, 15, 16, 23, 42]
print("STEP 1: a list")
print(f"  nums = {nums}   (length {len(nums)})")

# STEP 2: a for loop visits every item in turn.
running_total = 0
for n in nums:
    running_total += n
print("")
print("STEP 2: sum with a loop")
print(f"  total = {running_total}")

# STEP 3: a comprehension builds a new list from an old one, in one line.
doubled = [n * 2 for n in nums]
evens = [n for n in nums if n % 2 == 0]
print("")
print("STEP 3: transform and filter with comprehensions")
print(f"  doubled = {doubled}")
print(f"  evens   = {evens}")

# STEP 4: invariants. The sum matches Python's own sum(), and doubling then
# halving returns the original list. If either fails, the logic is wrong.
ok = (running_total == sum(nums)) and ([d // 2 for d in doubled] == nums)
print("")
print(f"STEP 4: loop-sum matches sum(), and double-then-halve round-trips: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Lists and loops are the workhorse of ML code. Next: functions and dictionaries.")
