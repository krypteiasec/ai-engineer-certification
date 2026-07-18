#!/usr/bin/env python3
"""
LAB AE3: Structured output in production, with schema validation and repair.

When another program consumes a model's answer, prose is a liability and JSON is
the contract. But a contract you do not enforce is a wish. Two disciplines make
structured output safe in production. You VALIDATE every response against a
schema (required fields, right types) before you trust it. And when validation
fails, you REPAIR: retry with a firmer instruction rather than crashing or
shipping garbage downstream. In this lab a first, weak call returns text that is
not JSON, validation catches it, and a repair retry with an explicit extract
instruction returns output that passes the schema.

Run: python3 modules/academy-content/labs/app-engineering/ae3-structured-output.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


# The schema the downstream system requires: these fields, these types.
SCHEMA = {"name": str, "number": (int, float), "email": str}


def validate(obj, schema):
    """Return (ok, reason). A response is only trustworthy if it parses AND
    matches the schema, so this runs before any downstream use."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    for key, typ in schema.items():
        if key not in obj:
            return False, "missing field: " + key
        if not isinstance(obj[key], typ):
            return False, "wrong type for field: " + key
    return True, "ok"


def try_parse(text):
    """Parse defensively. A model can return prose, half-JSON, or fenced JSON;
    your code must never crash on it."""
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def attempt_call(source, firm):
    """First pass (firm=False) is a weak instruction that does not force JSON.
    The repair pass (firm=True) uses an explicit extract-JSON instruction. Same
    idea as re-prompting a real model 'return ONLY valid JSON' on a bad reply."""
    if firm:
        return complete("Extract JSON from this text: " + source)
    return complete("Summarize this: " + source)


source = "my name is Ada and the number is 42 and email is ada@example.com"

final = None
used_repair = False
for attempt in range(2):
    firm = (attempt == 1)
    raw = attempt_call(source, firm)
    print("STEP %d: %s call" % (attempt + 1, "repair (firm)" if firm else "first (weak)"))
    print("  raw output :", repr(raw))
    parsed_ok, obj = try_parse(raw)
    if not parsed_ok:
        print("  parse      : FAILED (not JSON) -> will repair")
        used_repair = True
        continue
    valid_ok, reason = validate(obj, SCHEMA)
    if not valid_ok:
        print("  schema     : FAILED (%s) -> will repair" % reason)
        used_repair = True
        continue
    print("  schema     : PASSED")
    final = obj
    break

print("")
print("STEP 3: result")
print("  final object :", final)
print("  needed repair:", used_repair)

valid_final = final is not None and validate(final, SCHEMA)[0]
fields_ok = valid_final and final.get("name") == "Ada" and final.get("number") == 42

ok = valid_final and fields_ok and used_repair
print("")
print("  final passed schema validation :", valid_final)
print("  fields extracted correctly      :", fields_ok)
print("  repair path was exercised       :", used_repair)
print("")
print("SCHEMA REPAIR RECOVERED INVALID OUTPUT: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Validate, then repair, then trust. Next: make the calls survive failure.")
