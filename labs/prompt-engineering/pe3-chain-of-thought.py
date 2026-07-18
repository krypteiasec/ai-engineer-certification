#!/usr/bin/env python3
"""
LAB PE3: Chain-of-thought and decomposition.

Hard questions have steps. Ask a model to jump straight to the final answer and
it grabs the nearest-looking fact and gets it wrong. Break the question into
ordered sub-questions, answer each, and chain the results, and it lands the right
answer. In this lab you ask a two-hop question one-shot (wrong) and then
decomposed step by step (right), and prove the step-by-step path reaches the
correct answer the one-shot path misses.

Run: python3 modules/academy-content/labs/prompt-engineering/pe3-chain-of-thought.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

FACTS = ("Alice is the manager. "
         "The manager approves budgets. "
         "Budgets over 1000 need a review.")

# The real answer to "who approves budgets over 1000" is Alice: manager approves
# budgets, and Alice is the manager. That takes two hops.

# 1. ONE-SHOT: ask for the final answer directly. The model grabs the sentence
#    that shares the most words with the question, which is the wrong one.
one_shot = complete(f"Context: {FACTS} Question: Who approves budgets over 1000?")
print("STEP 1: one-shot, jump to the answer")
print(f"  answer : {one_shot!r}")
one_shot_right = "alice" in one_shot.lower()

# 2. DECOMPOSE: break it into ordered sub-questions and answer each from context.
print("")
print("STEP 2: decompose into steps and chain the results")
step_a = complete(f"Context: {FACTS} Question: Who is the manager?")
print(f"  step a -> who is the manager?   : {step_a!r}")
step_b = complete(f"Context: {FACTS} Question: Who approves budgets?")
print(f"  step b -> who approves budgets? : {step_b!r}")

# Chain: step b says the manager approves; step a says the manager is Alice.
chained = step_a if ("manager" in step_b.lower()) else step_b
print(f"  chain  -> the approver is       : {chained!r}")
decomposed_right = "alice" in chained.lower()

# 3. Step-by-step reached Alice; the one-shot shortcut did not.
print("")
print(f"STEP 3: decomposed reached the right answer : {decomposed_right}")
print(f"        one-shot reached the right answer   : {one_shot_right}")

ok = decomposed_right and not one_shot_right
print("")
print(f"STEP-BY-STEP BEATS ONE-SHOT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Decomposition is reasoning made explicit. Next: structured JSON output.")
