#!/usr/bin/env python3
"""
LAB LO7: The Pulse dashboard, module registry and health check.

Pulse is the Life Dashboard, the single surface where you see your state, goals,
and work. It is one daemon that registers a set of modules and answers a health
probe so you know it is alive. In this lab you build a sample Pulse: register
modules, run the health check, and prove every declared module came up and the
dashboard reports healthy. No server and no network; it is a fictional in-memory
model of the real /api/pulse/health probe.

Run: python3 modules/academy-content/labs/lifeos/lo7-pulse-health.py
"""
import sys

# STEP 1: a sample Pulse daemon that modules register into.
class Pulse:
    def __init__(self):
        self.modules = {}
    def register(self, name):
        self.modules[name] = "up"
    def health(self):
        # healthy only when every registered module reports up.
        all_up = all(state == "up" for state in self.modules.values())
        return {"ok": all_up, "modules": len(self.modules)}

declared = ["observability", "hooks", "wiki", "voice", "telos"]
pulse = Pulse()
for m in declared:
    pulse.register(m)

print("STEP 1: registered Pulse modules")
for m in sorted(pulse.modules):
    print(f"  {m:16} -> {pulse.modules[m]}")

# STEP 2: run the health probe (the real dashboard answers this on :31337).
h = pulse.health()
print("")
print(f"STEP 2: health probe -> ok={h['ok']}  modules={h['modules']}")

# STEP 3: invariants. every declared module registered, and the probe is healthy.
all_registered = set(pulse.modules) == set(declared)
ok = all_registered and h["ok"] and h["modules"] == len(declared)
print("")
print(f"PULSE HEALTHY, ALL MODULES UP: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("One daemon, all your modules, one health probe. Next: compose it into a daily OS.")
