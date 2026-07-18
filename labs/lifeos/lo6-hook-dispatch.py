#!/usr/bin/env python3
"""
LAB LO6: A hook firing on a lifecycle event.

Hooks are how LifeOS acts without being asked. The harness emits lifecycle events
(SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop, SessionEnd, and
more), and each event runs the hooks registered to it. In this lab you build a
tiny event bus, register sample hooks against events, fire a session, and prove
that firing SessionStart runs exactly its registered hooks and nothing else. All
handlers are harmless stand-ins that just record that they ran.

Run: python3 modules/academy-content/labs/lifeos/lo6-hook-dispatch.py
"""
import sys

# STEP 1: a registry mapping each event to its ordered list of hooks.
ran = []                                    # global record of what fired

def load_context():   ran.append("LoadContext")
def scan_prompt():    ran.append("PromptGuard")
def track_tool():     ran.append("ToolActivityTracker")
def capture_learn():  ran.append("WorkCompletionLearning")

registry = {
    "SessionStart":     [load_context],
    "UserPromptSubmit": [scan_prompt],
    "PreToolUse":       [track_tool],
    "Stop":             [capture_learn],
}

def fire(event):
    fired = []
    for hook in registry.get(event, []):    # unknown event -> nothing fires
        hook()
        fired.append(hook.__name__)
    return fired

# STEP 2: fire a realistic slice of a session.
print("STEP 1-2: fire lifecycle events")
for ev in ["SessionStart", "UserPromptSubmit", "PreToolUse", "Stop"]:
    fired = fire(ev)
    print(f"  {ev:18} -> {fired}")

# STEP 3: an event with no registered hooks fires nothing (no surprises).
none_fired = fire("PreCompact")
print(f"  {'PreCompact':18} -> {none_fired}  (no hooks registered)")

# INVARIANT: SessionStart ran exactly LoadContext, the total fired count matches
# the four registered events, and the unregistered event was a clean no-op.
sessionstart_only_loadcontext = ran[:1] == ["LoadContext"]
total_ok = ran == ["LoadContext", "PromptGuard", "ToolActivityTracker", "WorkCompletionLearning"]
noop_ok = none_fired == []
ok = sessionstart_only_loadcontext and total_ok and noop_ok
print("")
print(f"HOOKS FIRE ON THEIR EVENTS ONLY (registered fire, unknown no-ops): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Events drive hooks; hooks drive automation. Next: where you see it all, the Pulse dashboard.")
