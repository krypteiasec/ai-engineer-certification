#!/usr/bin/env python3
"""
LAB PE4: Structured and JSON output.

Prose is for humans. Software needs structure it can parse. When you ask a model
for JSON, you get output another program can load, validate, and act on, instead
of a paragraph you have to scrape. In this lab you extract facts as JSON, load it
with a real json.loads, and prove it parses into the exact fields you expected,
while a free-text answer to the same content does not parse at all.

Run: python3 modules/academy-content/labs/prompt-engineering/pe4-structured-output.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine, tool_route

text = "My name is Ada and the number is 42 and my email is ada@example.com"

# 1. FREE TEXT: no structure asked for, so the reply is not machine-parseable.
free = complete(text)
print("STEP 1: free-text prompt (no structure)")
print(f"  output : {free!r}")
try:
    json.loads(free)
    free_parses = True
except (json.JSONDecodeError, TypeError):
    free_parses = False

# 2. STRUCTURED: ask for JSON. Now the output is a real object with named fields.
structured = complete("Extract JSON from this: " + text)
print("")
print("STEP 2: ask for JSON")
print(f"  output : {structured!r}")
obj = json.loads(structured)          # a real parse; raises if it is not JSON
print(f"  parsed : name={obj.get('name')!r} number={obj.get('number')!r} email={obj.get('email')!r}")

# 3. The structured output parses into the exact fields; the free text does not.
fields_ok = (obj.get("name") == "Ada" and obj.get("number") == 42
             and obj.get("email") == "ada@example.com")
print("")
print(f"STEP 3: JSON parsed into expected fields : {fields_ok}")
print(f"        free text was parseable JSON     : {free_parses}")

ok = fields_ok and not free_parses
print("")
print(f"OUTPUT PARSES AS JSON: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Ask for structure and software can trust the output. Next: XML tagging.")
