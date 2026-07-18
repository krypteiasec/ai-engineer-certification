#!/usr/bin/env python3
"""
LAB SDK8: Build a small app with the SDK.

Now assemble everything into one working app. A review-triage service takes a
customer review, uses the model to classify it, returns a STRUCTURED record your
database can store, caches the stable system prompt across reviews, and stays
inside a token budget. It uses every piece from this course at once:

  - a Client wrapping client.messages.create (lab SDK1)
  - a system prompt that sets the standing job (lab SDK2)
  - prompt caching so the shared instructions bill once (lab SDK7)
  - structured, validated JSON output per review (lab SDK6)
  - token accounting against a budget (lab SDK7)

Where would the real Claude Agent SDK fit? If the job needed to read and edit
files, run bash, and search the web autonomously, you would reach for the Claude
Agent SDK (claude-agent-sdk), which ships that harness and those built-in tools.
For a bounded classify-and-return service like this, a plain Messages API client
is exactly the right, simplest tier. Choosing the smallest tier that does the job
is the real skill. In this lab you run the app over two reviews and prove it
returns correct structured records, bills the cached prefix once, and stays under
budget.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk8-build-app.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


def count_tokens(text):
    return len(text.split())


SYSTEM = ("You are a review triage service. Classify each review's sentiment as "
          "positive, negative, or neutral, and return only the label. " * 5)   # stable, cacheable

RECORD_SCHEMA = {"sentiment": str, "priority": str}


class TriageApp:
    """The whole app behind one method. Caches the system prefix after the first
    call, classifies each review, and emits a validated {sentiment, priority}
    record. Tracks billed tokens so we can hold a budget."""
    def __init__(self, model="claude-opus-4-8"):
        self.model = model
        self._cached = False
        self.total_billed = 0.0

    def _bill(self, system, question):
        sys_t, q_t = count_tokens(system), count_tokens(question)
        if not self._cached:
            self._cached = True
            self.total_billed += q_t + 1.25 * sys_t      # first call writes the cache
        else:
            self.total_billed += q_t + 0.1 * sys_t       # later calls read the cache

    def triage(self, review):
        # Classify with the model (system sets the job; review is the input).
        label = complete(SYSTEM + "\nsentiment: " + review)
        self._bill(SYSTEM, review)
        # A negative review is high priority; anything else is normal. Return a
        # structured record, exactly like an output_config.format response.
        priority = "high" if label == "negative" else "normal"
        record = {"sentiment": label, "priority": priority}
        return json.dumps(record, sort_keys=True)


def validate(obj, schema):
    for field, typ in schema.items():
        if field not in obj or not isinstance(obj[field], typ):
            return False
    return True


# --- Run the app over a batch of reviews. ------------------------------------
REVIEWS = ["I love this, it works perfectly", "the update is terrible and buggy"]
BUDGET = 400   # a soft token budget the batch must stay under

app = TriageApp()
records = []
print("STEP 1: triage each review into a structured record")
for r in REVIEWS:
    raw = app.triage(r)
    obj = json.loads(raw)
    records.append(obj)
    print("  %-40s -> %s" % (repr(r), raw))

print("")
print("STEP 2: app-level accounting")
print("  total billed tokens :", "%.1f" % app.total_billed, "(budget %d)" % BUDGET)
print("  cache engaged        :", app._cached)

all_valid = all(validate(rec, RECORD_SCHEMA) for rec in records)
correct = (records[0]["sentiment"] == "positive" and records[0]["priority"] == "normal"
           and records[1]["sentiment"] == "negative" and records[1]["priority"] == "high")
under_budget = (app.total_billed < BUDGET)
cache_used = (app._cached is True)

print("")
print("STEP 3: checks")
print("  every record matches the schema        :", all_valid)
print("  sentiment and priority are correct     :", correct)
print("  the batch stayed under the token budget:", under_budget)
print("  the cached prefix was engaged          :", cache_used)

ok = all_valid and correct and under_budget and cache_used
print("")
print("THE SDK APP ANSWERED CORRECTLY END TO END: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("You can build with the Claude API and SDK. Course complete.")
