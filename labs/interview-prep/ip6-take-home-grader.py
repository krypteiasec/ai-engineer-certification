#!/usr/bin/env python3
"""
LAB IP6: A mock take-home grader.

Take-home projects are graded against a rubric, and the strongest thing you can
do is grade your OWN submission the way the reviewer will before you send it.
This lab flips to the reviewer's seat: given a spec (a set of weighted checks)
and a candidate implementation, run the checks and produce a score. It makes the
grading criteria concrete, which is exactly the mindset that passes take-homes:
build to the checks, not to a vibe.

The spec here grades a `tokenize` function on correctness, edge-case handling,
and hygiene (a docstring). You PROVE:
  - a good candidate scores full marks,
  - a buggy candidate scores strictly lower and fails the specific checks it
    should fail,
  - the grader is deterministic and the good submission outranks the buggy one.

Run: python3 modules/academy-content/labs/interview-prep/ip6-take-home-grader.py
"""
import sys


# ---- Two candidate submissions to the same spec ---------------------------
def good_tokenize(text):
    """Lowercase and split on whitespace, returning a list of tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.lower().split()


def buggy_tokenize(text):
    # No docstring, no type guard, and it forgets to lowercase.
    return text.split()


# ---- The rubric: weighted, self-checking criteria -------------------------
def build_spec():
    """Each check returns True/False for a candidate. Weights sum to 1.0 so the
    score is a clean fraction, which is how a real rubric reports."""
    def basic_split(fn):
        return fn("the cat sat") == ["the", "cat", "sat"]

    def lowercases(fn):
        return fn("The CAT Sat") == ["the", "cat", "sat"]

    def handles_empty(fn):
        try:
            return fn("") == []
        except Exception:
            return False

    def guards_type(fn):
        try:
            fn(123)
            return False  # should have raised
        except TypeError:
            return True
        except Exception:
            return False

    def has_docstring(fn):
        return bool(fn.__doc__ and fn.__doc__.strip())

    return [
        ("splits on whitespace", basic_split, 0.30),
        ("lowercases input", lowercases, 0.25),
        ("handles empty string", handles_empty, 0.15),
        ("guards bad input type", guards_type, 0.20),
        ("documents the function", has_docstring, 0.10),
    ]


def grade(candidate, spec):
    """Run every check against the candidate, return (score, per-check results)."""
    results = []
    score = 0.0
    for name, check, weight in spec:
        try:
            passed = bool(check(candidate))
        except Exception:
            passed = False
        if passed:
            score += weight
        results.append((name, passed, weight))
    return round(score, 4), results


spec = build_spec()
good_score, good_results = grade(good_tokenize, spec)
buggy_score, buggy_results = grade(buggy_tokenize, spec)

print("STEP 1: grade the good submission")
for name, passed, weight in good_results:
    print(f"  [{'x' if passed else ' '}] {name:24s} (weight {weight:.2f})")
print(f"  score = {good_score}")

print("")
print("STEP 2: grade the buggy submission")
for name, passed, weight in buggy_results:
    print(f"  [{'x' if passed else ' '}] {name:24s} (weight {weight:.2f})")
print(f"  score = {buggy_score}")

# The buggy one must fail exactly: lowercasing, type guard, and docstring.
buggy_failed = {name for name, passed, _ in buggy_results if not passed}
expected_failures = {"lowercases input", "guards bad input type", "documents the function"}

good_full = (good_score == 1.0)
buggy_lower = (buggy_score < good_score)
buggy_expected = (buggy_failed == expected_failures)
deterministic = (grade(good_tokenize, spec)[0] == good_score)

print("")
print(f"        good scores full marks       : {good_full}")
print(f"        buggy scores strictly lower   : {buggy_lower} ({buggy_score} < {good_score})")
print(f"        buggy fails the right checks  : {buggy_expected}")
print(f"        grader is deterministic       : {deterministic}")

ok = good_full and buggy_lower and buggy_expected and deterministic
print("")
print(f"TAKE-HOME GRADER SCORES THE CANDIDATE CORRECTLY: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Grade yourself before they do. Next round: behavioral and safety.")
