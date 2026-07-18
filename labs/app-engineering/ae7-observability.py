#!/usr/bin/env python3
"""
LAB AE7: Observability. Structured tracing for LLM apps.

When an LLM feature misbehaves in production, you cannot debug what you did not
record. Observability means every request emits a STRUCTURED trace: an id, the
model, input and output tokens, latency, status, and a timestamp, as machine
-readable records you can query and aggregate. From those records you get the
metrics that run the service: total tokens (cost), p50 latency (speed), and error
count (reliability). In this lab a tracer wraps each call, including one that
errors, and proves it captures exactly one structured record per request plus
the rolled-up metrics.

Run: python3 modules/academy-content/labs/app-engineering/ae7-observability.py
"""
import sys, os, json
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine

REQUIRED_FIELDS = {"id", "model", "status", "input_tokens", "output_tokens", "latency_ms", "ts_ms"}


class Tracer:
    """Wraps a provider call and records one structured trace per request. The
    clock is a deterministic counter (not wall time) so the lab is reproducible;
    in production this is a real monotonic clock feeding your logs and dashboards."""
    def __init__(self):
        self.records = []
        self._clock = 0.0
        self._id = 0

    def traced(self, model, caller, prompt):
        self._id += 1
        rid = "req-%03d" % self._id
        start = self._clock
        status = "ok"
        out = ""
        try:
            out = caller(prompt)
        except Exception:
            status = "error"
            out = ""
        # Deterministic simulated latency: a base plus a per-output-token cost.
        latency = 5.0 + 2.0 * len(out.split())
        self._clock += latency
        rec = {
            "id": rid,
            "model": model,
            "status": status,
            "input_tokens": len(prompt.split()),
            "output_tokens": len(out.split()),
            "latency_ms": round(latency, 2),
            "ts_ms": round(start, 2),
        }
        self.records.append(rec)
        return out, rec

    def as_jsonl(self):
        return "\n".join(json.dumps(r, sort_keys=True) for r in self.records)

    def metrics(self):
        total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in self.records)
        errors = sum(1 for r in self.records if r["status"] == "error")
        lat = sorted(r["latency_ms"] for r in self.records)
        p50 = lat[len(lat) // 2] if lat else 0.0
        return {"requests": len(self.records), "total_tokens": total_tokens,
                "errors": errors, "p50_latency_ms": p50}


def boom(_prompt):
    raise RuntimeError("provider unavailable")


tracer = Tracer()

print("STEP 1: run traced requests (one intentionally errors)")
tracer.traced("academy-mini", complete, "Classify the sentiment. Input: I love this product")
tracer.traced("academy-mini", complete, "Extract JSON: my name is Ada and the number is 42")
tracer.traced("academy-large", complete,
              "Context: The moon orbits the earth. Question: What orbits the earth?")
tracer.traced("academy-mini", boom, "Classify the sentiment. Input: this will fail")

print("")
print("STEP 2: the structured trace log (one JSON record per request)")
print(tracer.as_jsonl())

m = tracer.metrics()
print("")
print("STEP 3: rolled-up metrics")
print("  requests        :", m["requests"])
print("  total tokens    :", m["total_tokens"])
print("  errors          :", m["errors"])
print("  p50 latency ms  :", m["p50_latency_ms"])

one_per_request = (len(tracer.records) == 4)
all_fields = all(REQUIRED_FIELDS.issubset(r.keys()) for r in tracer.records)
caught_error = (m["errors"] == 1)
tokens_counted = (m["total_tokens"] > 0)

print("")
print("  one record per request     :", one_per_request)
print("  every record fully fielded :", all_fields)
print("  the error was recorded     :", caught_error)
print("  tokens were accounted      :", tokens_counted)

ok = one_per_request and all_fields and caught_error and tokens_counted
print("")
print("TRACING CAPTURED ONE STRUCTURED RECORD PER REQUEST: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Record every request as structured data, then the metrics run the service. Next: assemble the whole app.")
