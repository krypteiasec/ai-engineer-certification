#!/usr/bin/env python3
"""
LAB PE6: Prompt caching and cost.

You pay per token. Most apps resend the SAME long system prompt on every single
request, paying full price for it every time. Prompt caching stores that stable
prefix after the first call so later calls only pay for the new part. In this lab
you run one stable system prompt across several queries, count the tokens billed
with and without caching, and prove caching cuts the repeated cost while the
answers stay identical.

Run: python3 modules/academy-content/labs/prompt-engineering/pe6-prompt-caching.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

def ntokens(s):
    return len(s.split())

# A big stable system prompt reused on every request (the expensive, repeated part).
SYSTEM = ("You are a careful sentiment classifier. " * 12).strip()
queries = [
    "Classify the sentiment: I love this",
    "Classify the sentiment: this is terrible",
    "Classify the sentiment: it is good and nice",
    "Classify the sentiment: a broken awful mess",
]
sys_tok = ntokens(SYSTEM)
print(f"stable system prompt = {sys_tok} tokens, reused across {len(queries)} queries")
print("")

# 1. NO CACHE: every request pays for the full system prompt + its query.
no_cache = 0
answers_a = []
for q in queries:
    no_cache += sys_tok + ntokens(q)
    answers_a.append(complete(SYSTEM + "\n" + q))
print(f"STEP 1: no cache, tokens billed = {no_cache}")

# 2. CACHE: pay the system prompt ONCE, then only the per-query tokens after that.
cache = sys_tok
answers_b = []
for q in queries:
    cache += ntokens(q)          # only the new suffix is billed on a cache hit
    answers_b.append(complete(SYSTEM + "\n" + q))
print(f"STEP 2: with cache, tokens billed = {cache}")

saved = no_cache - cache
print("")
print(f"STEP 3: tokens saved = {saved} ({100 * saved // no_cache}% cheaper)")

# 4. Caching must not change the answers, only the cost.
same = (answers_a == answers_b)
ok = (cache < no_cache) and same
print("")
print(f"        answers identical with and without cache : {same}")
print(f"PROMPT CACHING CUTS REPEATED TOKENS: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Cache the stable prefix, pay for it once. Next: context engineering.")
