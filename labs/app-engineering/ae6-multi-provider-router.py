#!/usr/bin/env python3
"""
LAB AE6: Multi-provider architecture. Route by task, fall back on outage.

Betting a product on one provider is a single point of failure and usually the
wrong cost curve. The pattern is a router behind one interface: send cheap,
simple tasks to a small fast provider and hard tasks to a capable one, and when
the chosen provider is down, fall back to the other so the request still gets
answered. In this lab two providers wrap the shared model. The router sends a
simple task to the fast provider, then sends a complex task to the smart
provider, which is down, and proves the request falls back and still succeeds.

Run: python3 modules/academy-content/labs/app-engineering/ae6-multi-provider-router.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


class Provider:
    """One provider behind a uniform interface. `up=False` simulates an outage
    so the router's fallback path is provable offline."""
    def __init__(self, name, up=True):
        self.name = name
        self.up = up
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if not self.up:
            raise RuntimeError("%s is down" % self.name)
        return complete(prompt)


def classify_task(prompt):
    """Cheap heuristic router key: extraction/structured work is 'complex' and
    goes to the capable model; everything else is 'simple' and goes to the fast
    one. Real routers score by task, length, and required quality."""
    low = prompt.lower()
    if "extract" in low or "json" in low:
        return "complex"
    return "simple"


class Router:
    def __init__(self, fast, smart):
        self.fast = fast    # cheap, for simple tasks
        self.smart = smart  # capable, for complex tasks

    def route(self, prompt):
        task = classify_task(prompt)
        primary = self.smart if task == "complex" else self.fast
        backup = self.fast if primary is self.smart else self.smart
        try:
            return task, primary.name, primary.complete(prompt)
        except RuntimeError as e:
            print("  primary %s failed (%s) -> falling back to %s" % (primary.name, e, backup.name))
            return task, backup.name + " (fallback)", backup.complete(prompt)


fast = Provider("fast-mini", up=True)
smart = Provider("smart-large", up=False)   # the capable provider is having an outage
router = Router(fast, smart)

print("STEP 1: a simple task routes to the fast provider")
task1, who1, ans1 = router.route("Classify the sentiment. Input: I love this product")
print("  task=%s served-by=%s answer=%r" % (task1, who1, ans1))

print("")
print("STEP 2: a complex task routes to the smart provider, which is DOWN")
task2, who2, ans2 = router.route("Extract JSON: my name is Ada and the number is 42")
print("  task=%s served-by=%s answer=%r" % (task2, who2, ans2))

routed_by_task = (task1 == "simple" and who1 == "fast-mini"
                  and task2 == "complex")
fell_back = who2.endswith("(fallback)") and ans2 != ""
both_answered = (ans1 == "positive") and ('"name": "Ada"' in ans2)

print("")
print("STEP 3: result")
print("  routed each task to the right provider :", routed_by_task)
print("  fell back when the primary was down    :", fell_back)
print("  both requests still got an answer      :", both_answered)

print("")
print("ROUTER ROUTED BY TASK: %s" % ("YES" if routed_by_task else "NO"))
ok = routed_by_task and fell_back and both_answered
print("ROUTER FELL BACK ON PROVIDER OUTAGE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("One interface, many providers, route by task, fall back on failure. Next: see all of it with tracing.")
