#!/usr/bin/env python3
"""
LAB EV6: pass@k and statistical literacy.

A language model is stochastic: run the same prompt twice with temperature above
zero and you can get a pass one time and a fail the next. Scoring one sample and
calling it done is how you fool yourself. pass@k measures the probability that at
least one of k samples passes, and because more tries can only help, pass@k rises
with k. In this lab you sample a stochastic system under test across many seeds,
estimate pass@1 (a single try) and pass@3 (best of three), and prove pass@3 is
strictly higher. That is why an honest eval reports pass@k, not one lucky run.

Run: python3 modules/academy-content/labs/evals/ev6-pass-at-k.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

# A generation task with enough vocabulary that sampling at temperature > 0 does
# NOT always include the required word. "pass" = the target word appears.
QUERY = ("Complete this list: apple banana cherry date fig grape lemon mango "
         "olive peach plum melon")
TARGET = "apple"

# The stochastic system under test: same prompt, different seed, different sample.
def sample(seed):
    out = complete(QUERY, temperature=0.7, seed=seed)
    return TARGET in out.split()

M = 24
results = [sample(s) for s in range(M)]        # one draw per seed, deterministic set
passes = sum(results)

# pass@1: the single-try pass rate, estimated over all samples.
pass_at_1 = passes / M

# pass@3: split the draws into non-overlapping groups of 3, a group passes if ANY
# of its three samples passes. This is best-of-three.
K = 3
groups = [results[i:i + K] for i in range(0, M, K)]
pass_at_3 = sum(1 for g in groups if any(g)) / len(groups)

print(f"STEP 1: draw {M} samples from the stochastic system under test")
print(f"  individual passes : {passes}/{M}")
print(f"  at least one fail : {passes < M}")
print(f"  at least one pass : {passes > 0}")
print("")
print("STEP 2: estimate pass@1 and pass@3")
print(f"  pass@1 (one try)      : {pass_at_1:.2f}")
print(f"  pass@3 (best of three): {pass_at_3:.2f}")

# The point: pass@3 must be strictly greater than pass@1, and the SUT must be
# genuinely stochastic (some pass, some fail) or the metric is meaningless.
stochastic = (0 < passes < M)
ok = stochastic and (pass_at_3 > pass_at_1)
print("")
print(f"PASS@K RISES WITH K (pass@1 < pass@3): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("One run lies; pass@k tells the truth. Next: gate releases on a regression.")
