#!/usr/bin/env python3
"""
LAB CF7: Context management and reliability (Domain 5, 15%).

The smallest domain, but its failure modes cascade. This lab proves three
patterns the exam tests. Front-loading beats burying: a fact placed at the top
of a long context survives, one buried in the middle is at risk (lost in the
middle). A persisted structured facts block keeps exact values that progressive
summarization would blur. And a PostToolUse trim drops 35 of 40 noisy fields so
the model focuses on what matters.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf7-context-reliability.py
"""
import sys

# STEP 1: lost in the middle. Model reliably reads the head and tail of a long
# context; a fact in the deep middle is unreliable. Front-load key findings.
def position_reliable(index: int, length: int) -> bool:
    """Head (first 10%) and tail (last 10%) are reliable; deep middle is not."""
    head = index <= length * 0.1
    tail = index >= length * 0.9
    return head or tail

LEN = 100
buried = position_reliable(50, LEN)     # middle: unreliable
frontloaded = position_reliable(2, LEN) # top: reliable
print("STEP 1: lost in the middle")
print(f"  fact buried at position 50/100 reliable? {buried}")
print(f"  same fact front-loaded at 2/100 reliable? {frontloaded}")
position_ok = (buried is False and frontloaded is True)

# STEP 2: persisted facts block vs progressive summarization. Exact values must
# survive compaction, so keep them in a structured block, not in prose history.
CASE_FACTS = {
    "customer_id": "CUST-12345",
    "order_id": "ORD-67890",
    "order_total": 89.99,
    "order_date": "2025-01-15",
}


def progressive_summary(total: float) -> str:
    return f"about ${round(total / 10) * 10}"   # 89.99 -> 'about $90', value lost


def persisted_block(facts: dict) -> float:
    return facts["order_total"]                 # exact value preserved

summarized = progressive_summary(CASE_FACTS["order_total"])
preserved = persisted_block(CASE_FACTS)
print("")
print("STEP 2: exact values survive only in a persisted facts block")
print(f"  progressive summary -> {summarized} (exact value lost)")
print(f"  persisted block     -> ${preserved} (exact value kept)")
facts_ok = (summarized == "about $90" and preserved == 89.99)

# STEP 3: PostToolUse trim. A tool returns 40 fields; the task needs 5. Trim the
# rest before it reaches the model.
NEEDED = ["order_id", "status", "total", "items", "return_eligible"]
raw_result = {f"field_{i}": i for i in range(40)}
raw_result.update({k: k for k in NEEDED})


def post_tool_use_trim(result: dict, keep: list) -> dict:
    return {k: result[k] for k in keep if k in result}

trimmed = post_tool_use_trim(raw_result, NEEDED)
print("")
print("STEP 3: PostToolUse trims a noisy tool result")
print(f"  raw fields: {len(raw_result)}  ->  trimmed fields: {len(trimmed)}")
trim_ok = (len(raw_result) >= 40 and set(trimmed.keys()) == set(NEEDED))

ok = position_ok and facts_ok and trim_ok
print("")
print(f"  front-loading beats burial (lost-in-middle) : {position_ok}")
print(f"  persisted block preserves exact values      : {facts_ok}")
print(f"  PostToolUse trim removes context noise      : {trim_ok}")
print("")
print(f"CONTEXT RELIABILITY RULES HOLD (front-load + persist facts + trim): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Put key findings at the top, keep exact values in a block, trim noisy results. Next: scenarios and exam day.")
