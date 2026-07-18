#!/usr/bin/env python3
"""
LAB SDK6: Structured output you can trust.

Prose is for humans. Software needs structure it can load. If your app stores,
routes, or passes a model's answer to another function, a paragraph is a scraping
problem waiting to break. The Messages API can CONSTRAIN the output to a JSON
schema so the result is guaranteed parseable:

    schema = {
        "type": "object",
        "properties": {
            "name":   {"type": "string"},
            "number": {"type": "integer"},
            "email":  {"type": "string"},
        },
        "required": ["name", "number"],
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": "Extract from: I am Ada, number 42, ada@x.com"}],
    )

Two notes. The canonical parameter is output_config.format; the older top-level
output_format is deprecated. The SDK also offers client.messages.parse() which
validates the response against the schema for you. And strict tool use
(strict: true on a tool definition) guarantees tool inputs validate the same way.
Even so, always wrap the parse in error handling: a model can still return
malformed output, and your code must fail safely rather than crash. In this lab
you extract facts as JSON, validate them against a schema, and prove a malformed
response is caught instead of crashing.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk6-structured-output.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


SCHEMA = {
    "type": "object",
    "properties": {
        "name":   {"type": "string"},
        "number": {"type": "integer"},
    },
    "required": ["name", "number"],
    "additionalProperties": False,
}


def validate(obj, schema):
    """A tiny schema check standing in for what output_config.format enforces on
    the server: required fields present, and each field the right JSON type."""
    py_type = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for field in schema.get("required", []):
        if field not in obj:
            return False, "missing required field: %s" % field
    for field, spec in schema.get("properties", {}).items():
        if field in obj and not isinstance(obj[field], py_type[spec["type"]]):
            return False, "wrong type for %s" % field
    return True, "ok"


def parse_safely(raw):
    """Always wrap the parse: a model CAN return malformed output. Return
    (ok, value_or_error) so the caller never crashes on bad JSON."""
    try:
        return True, json.loads(raw)
    except json.JSONDecodeError as e:
        return False, "malformed JSON: %s" % e


# --- 1. Constrained JSON output, parsed and validated. -----------------------
raw = complete("Extract JSON: my name is Ada and the number is 42")
print("STEP 1: the model returned structured JSON")
print("  raw      :", repr(raw))

parsed_ok, obj = parse_safely(raw)
valid_ok, why = validate(obj, SCHEMA) if parsed_ok else (False, obj)
print("  parsed   :", obj if parsed_ok else "(parse failed)")
print("  valid    :", valid_ok, "(%s)" % why)

# --- 2. A malformed response is caught, not crashed on. ----------------------
bad_raw = '{"name": "Ada", "number": 42'   # truncated: missing closing brace
bad_parsed_ok, bad_val = parse_safely(bad_raw)
print("")
print("STEP 2: a malformed response is handled safely")
print("  raw            :", repr(bad_raw))
print("  parse succeeded:", bad_parsed_ok, "->", bad_val)

fields_ok = parsed_ok and obj.get("name") == "Ada" and obj.get("number") == 42
schema_ok = valid_ok
malformed_caught = (bad_parsed_ok is False)   # we caught it, no exception escaped

print("")
print("STEP 3: checks")
print("  output parsed into the exact fields        :", fields_ok)
print("  it validated against the JSON schema       :", schema_ok)
print("  malformed output was caught, not crashed   :", malformed_caught)

ok = fields_ok and schema_ok and malformed_caught
print("")
print("STRUCTURED OUTPUT PARSES AND VALIDATES: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Constrain the shape, validate it, handle failure. Next: cut cost with caching.")
