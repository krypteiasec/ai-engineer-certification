#!/usr/bin/env python3
"""
LAB AE8: Putting it together. A small robust LLM app.

This is the whole course in one object. A production LLM feature is not a single
call, it is a pipeline: a client over the provider, a router that picks a
provider by task, retries with backoff for transient failures, fallback across
providers on an outage, a cost guard so spend stays bounded, and tracing so you
can see what happened. In this lab you assemble all of it into RobustLLMApp and
run a batch: one request retries through injected failures, one falls back when
its provider is down, all of them are cost-tracked and traced. If the whole
pipeline holds together, you can ship it.

Run: python3 modules/academy-content/labs/app-engineering/ae8-robust-app.py
"""
import sys, os, json, time
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


# --- Errors -----------------------------------------------------------------
class TransientError(Exception):
    """Retryable (a 429/503)."""


class BudgetExceeded(Exception):
    """The guard refused a call that would exceed the budget."""


# --- Cost (from AE5) --------------------------------------------------------
PRICES = {
    "fast-mini":   {"in": 0.0005, "out": 0.0015},
    "smart-large": {"in": 0.003,  "out": 0.015},
}


def estimate_cost(model, in_tokens, out_tokens):
    p = PRICES[model]
    return (in_tokens / 1000.0) * p["in"] + (out_tokens / 1000.0) * p["out"]


class CostGuard:
    def __init__(self, budget_usd):
        self.budget = budget_usd
        self.spent = 0.0

    def check(self, model, prompt, max_out=32):
        proj = estimate_cost(model, len(prompt.split()), max_out)
        if self.spent + proj > self.budget:
            raise BudgetExceeded("blocked: over budget $%.4f" % self.budget)

    def charge(self, model, prompt, out):
        self.spent += estimate_cost(model, len(prompt.split()), len(out.split()))


# --- Provider (from AE4/AE6): supports transient failures AND outage --------
class Provider:
    def __init__(self, name, up=True, fail_times=0):
        self.name = name
        self.up = up
        self.fail_times = fail_times
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if not self.up:
            raise RuntimeError("%s outage" % self.name)      # non-retryable here -> fallback
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TransientError("%s 429" % self.name)        # retryable -> backoff
        return complete(prompt)


# --- Tracer (from AE7) ------------------------------------------------------
class Tracer:
    def __init__(self):
        self.records = []
        self._clock = 0.0
        self._id = 0

    def traced(self, model, caller, prompt):
        self._id += 1
        start = self._clock
        status, out = "ok", ""
        try:
            out = caller(prompt)
        except Exception:
            status, out = "error", ""
        latency = 5.0 + 2.0 * len(out.split())
        self._clock += latency
        rec = {"id": "req-%03d" % self._id, "model": model, "status": status,
               "input_tokens": len(prompt.split()), "output_tokens": len(out.split()),
               "latency_ms": round(latency, 2), "ts_ms": round(start, 2)}
        self.records.append(rec)
        return out, rec


def classify_task(prompt):
    low = prompt.lower()
    return "complex" if ("extract" in low or "json" in low) else "simple"


# --- The assembled app ------------------------------------------------------
class RobustLLMApp:
    def __init__(self, fast, smart, budget_usd):
        self.fast = fast
        self.smart = smart
        self.guard = CostGuard(budget_usd)
        self.tracer = Tracer()
        self.retries = 0
        self.fallbacks = 0

    def _with_retry(self, provider, prompt, max_attempts=4):
        for _ in range(max_attempts):
            try:
                return provider.complete(prompt)
            except TransientError:
                self.retries += 1
                time.sleep(0.001)   # capped backoff, a few ms
        raise RuntimeError("%s exhausted retries" % provider.name)

    def ask(self, prompt):
        task = classify_task(prompt)
        primary = self.smart if task == "complex" else self.fast
        backup = self.fast if primary is self.smart else self.smart
        model = primary.name
        self.guard.check(model, prompt)          # cost gate BEFORE the call

        def pipeline(p):
            try:
                return self._with_retry(primary, p)      # retry transient failures
            except RuntimeError:                          # outage / exhausted -> fall back
                self.fallbacks += 1
                return self._with_retry(backup, p)

        out, rec = self.tracer.traced(model, pipeline, prompt)   # trace the whole thing
        self.guard.charge(model, prompt, out)
        return out, rec


# fast is flaky at first (2 transient failures); smart is fully down (forces a fallback).
fast = Provider("fast-mini", up=True, fail_times=2)
smart = Provider("smart-large", up=False)
app = RobustLLMApp(fast, smart, budget_usd=1.0)

batch = [
    "Classify the sentiment. Input: I love this product",          # simple -> fast (rides out 2 retries)
    "Extract JSON: my name is Ada and the number is 42",         # complex -> smart is DOWN -> fallback to fast
    "Classify the sentiment. Input: this update is terrible",      # simple -> fast (now healthy)
]

print("STEP 1: run the batch through the full pipeline")
answers = []
for p in batch:
    out, rec = app.ask(p)
    answers.append(out)
    print("  %s  model=%-11s status=%s  answer=%r" % (rec["id"], rec["model"], rec["status"], out))

print("")
print("STEP 2: what the pipeline did")
print("  requests traced :", len(app.tracer.records))
print("  retries         :", app.retries)
print("  fallbacks       :", app.fallbacks)
print("  total spend     : $%.6f of $%.2f budget" % (app.guard.spent, app.guard.budget))

all_answered = all(a for a in answers)
traced_all = (len(app.tracer.records) == 3)
retried = (app.retries == 2)
fell_back = (app.fallbacks == 1)
no_errors = all(r["status"] == "ok" for r in app.tracer.records)
within_budget = (0.0 < app.guard.spent <= app.guard.budget)

print("")
print("  every request answered      :", all_answered)
print("  all requests traced         :", traced_all)
print("  retry path exercised        :", retried)
print("  fallback path exercised     :", fell_back)
print("  no failed requests          :", no_errors)
print("  spend stayed within budget  :", within_budget)

ok = all_answered and traced_all and retried and fell_back and no_errors and within_budget
print("")
print("ROBUST APP RAN THE FULL PIPELINE END TO END: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Client, router, retries, fallback, cost guard, tracing: one pipeline that survives production.")
