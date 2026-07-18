#!/usr/bin/env python3
"""
LAB CF5: Prompt engineering and structured output (Domain 2, 20%).

Two exam rules dominate this domain. Nullable fields prevent hallucination
while required fields force it when the data is absent. And retries help for
format and structural errors but never conjure information that is genuinely
missing from the source. This lab uses the shared offline mock LLM to extract
JSON for real, then drills the schema decisions and a validation-retry loop, and
finishes with the Batch-vs-synchronous API router.

Deterministic, offline. Imports the shared academy_llm mock (standard library).
Run: python3 modules/academy-content/labs/cca-f/cf5-structured-output.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete

# STEP 1: structured output is parseable data, not prose. Ask the mock to
# extract JSON and load it with a real parser.
doc = "Extract JSON: the invoice name is Acme and the number is 5000"
raw = complete(doc)
obj = json.loads(raw)  # would raise if the output were not valid JSON
print("STEP 1: JSON extraction that a program can load")
print(f"  model output : {raw}")
print(f"  parsed name  : {obj.get('name')}   parsed number: {obj.get('number')}")
json_ok = obj.get("name") == "Acme" and obj.get("number") == 5000

# STEP 2: required vs nullable. Marking an ABSENT field as required forces the
# model to fabricate; a nullable field lets it return null honestly.
schema_required = {"required": ["due_date"]}          # bad: due_date absent in source
schema_nullable = {"nullable": ["due_date"]}          # good: allows null
source_has_due_date = False


def extract_field(present: bool, field: str, mode: str):
    if present:
        return "2025-01-15"
    if mode == "nullable":
        return None            # honest: not in the source
    return "FABRICATED-DATE"   # required forces a made-up value


req_val = extract_field(source_has_due_date, "due_date", "required")
null_val = extract_field(source_has_due_date, "due_date", "nullable")
print("")
print("STEP 2: required forces fabrication, nullable allows honest null")
print(f"  required due_date -> {req_val!r}  (fabricated, wrong)")
print(f"  nullable due_date -> {null_val!r}  (honest null, correct)")
nullable_ok = (req_val == "FABRICATED-DATE" and null_val is None)

# STEP 3: enums must carry 'other' and 'unclear' so data is never silently
# dropped.
ENUM = ["invoice", "receipt", "credit_note", "other", "unclear"]


def categorize(kind: str) -> str:
    return kind if kind in ENUM else "other"

enum_ok = ("other" in ENUM and "unclear" in ENUM and categorize("packing_slip") == "other")
print("")
print(f"STEP 3: enum {ENUM}")
print(f"  unknown 'packing_slip' -> {categorize('packing_slip')} (captured, not dropped)")

# STEP 4: validation-retry. Retry fixes a fixable format error; it cannot invent
# a value that is not in the source.
def validate(value):
    if value is None:
        return "value absent from source"
    if "/" in value:
        return "date not in ISO 8601 format"
    return None


def extract_with_retry(initial, source_present, max_retries=2):
    value = initial
    for _ in range(max_retries):
        err = validate(value)
        if not err:
            return value, "passed"
        if "format" in err:
            value = value.replace("03/05/2025", "2025-03-05")  # fixable, retry helps
        else:
            if not source_present:
                return value, "unfixable: data absent"  # retries cannot help
    return value, "exhausted"

fixed, status_fix = extract_with_retry("03/05/2025", source_present=True)
absent, status_absent = extract_with_retry(None, source_present=False)
print("")
print("STEP 4: validation-retry loop")
print(f"  bad-format date  -> {fixed} ({status_fix})")
print(f"  absent field     -> {absent} ({status_absent})")
retry_ok = (fixed == "2025-03-05" and status_fix == "passed"
            and status_absent == "unfixable: data absent")

# STEP 5: Batch API vs synchronous. If anyone is waiting, it is synchronous.
def api_for(scenario: str) -> str:
    waiting = scenario in ("pre-merge code review", "interactive support",
                           "ci/cd merge gate")
    return "synchronous" if waiting else "batch"

api_ok = (api_for("pre-merge code review") == "synchronous"
          and api_for("weekly security audit") == "batch")
print("")
print("STEP 5: Batch vs synchronous routing")
print(f"  pre-merge code review -> {api_for('pre-merge code review')}")
print(f"  weekly security audit -> {api_for('weekly security audit')}")

ok = json_ok and nullable_ok and enum_ok and retry_ok and api_ok
print("")
print(f"  JSON parses into named fields          : {json_ok}")
print(f"  nullable prevents fabrication          : {nullable_ok}")
print(f"  enum captures the unknown              : {enum_ok}")
print(f"  retry fixes format, not absence        : {retry_ok}")
print(f"  batch only when nobody is waiting      : {api_ok}")
print("")
print(f"STRUCTURED OUTPUT RULES HOLD: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Nullable over required, honest unclear over a wrong guess, sync when someone waits. Next: tool design and MCP.")
