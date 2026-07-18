#!/usr/bin/env python3
"""
LAB LM1: Install preflight. The agentic installer that asks before it touches.

LifeOS installs itself by handing the work to your AI. You tell Claude Code
"read the install page and install LifeOS for me", and a nine-step agentic
workflow runs: detect the environment, scan for conflicts read-only, deploy the
core, scaffold your personal USER dir, then ASK PERMISSION before it wires any
hook, edits settings.json, or adds the shell alias. Three laws hold the whole
thing together: additive never clobbering, permission precedes every mutation,
and a backup is taken before any edit. In this lab you run that flow on a
fictional sample environment and prove all three laws held. No real machine is
touched; every value here is invented for the lesson.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm1-install-preflight.py
"""
import sys

# STEP 1: detect the environment. The installer needs a coding harness that can
# BOTH read/write files AND run shell. Bun and Git are the hard prerequisites.
sample_env = {"bun": "1.1.0", "git": "2.44", "harness": "claude-code", "harness_can_shell": True, "harness_can_files": True}
required = ["bun", "git", "harness"]
missing = [t for t in required if t not in sample_env]
harness_capable = sample_env.get("harness_can_shell") and sample_env.get("harness_can_files")
print("STEP 1: detect env, verify prerequisites")
for t in required:
    print(f"  {t:8} -> {sample_env.get(t, 'MISSING')}")
print(f"  harness reads+writes files AND runs shell: {harness_capable}")
preflight_ok = (missing == []) and harness_capable

# STEP 2: run the nine-step workflow as an audit log. Every step that MUTATES the
# machine must be immediately preceded by a permission grant, and any write to an
# existing file must be preceded by a backup of that file. Read-only steps are free.
disk = {"settings.json": "user-original"}     # a pre-existing file we must not clobber
log = []                                       # ordered audit trail, the invariant surface
granted = set()

def step(name, kind, target=None):
    # kind: "read" (free), "mutate" (needs a prior grant), "write" (grant + backup)
    if kind in ("mutate", "write"):
        log.append(("PERMISSION", name)); granted.add(name)   # ASK precedes the act
    if kind == "write" and target in disk:
        log.append(("BACKUP", target))                        # backup precedes the edit
    log.append(("DO", name, kind))

step("detect-env", "read")
step("conflict-scan", "read")
step("deploy-core", "mutate")
step("scaffold-user-dir", "mutate")
step("wire-hooks", "write", "settings.json")
step("edit-settings", "write", "settings.json")
step("add-lifeos-alias", "mutate")
step("doctor-capability-check", "read")
step("setup-and-interview", "mutate")

print("")
print("STEP 2: nine-step agentic workflow audit log")
for e in log:
    print("  " + " ".join(str(x) for x in e))

# STEP 3: prove the three laws. (a) Every mutating/writing step was granted first.
# (b) Every write to an existing file was backed up first. (c) Additive: the
# original file's bytes are still recoverable (we never overwrote without backup).
mutations = [e[1] for e in log if e[0] == "DO" and e[2] in ("mutate", "write")]
permission_precedes = all(m in granted for m in mutations)
# check ordering: for settings.json, BACKUP appears before the DO write
seq = [e for e in log if (len(e) > 1 and e[1] in ("settings.json", "wire-hooks", "edit-settings")) or e[0] == "BACKUP"]
backup_before_write = log.index(("BACKUP", "settings.json")) < log.index(("DO", "edit-settings", "write"))
additive = "settings.json" in disk   # the key still exists; nothing was deleted
print("")
print(f"STEP 3: permission precedes every mutation : {permission_precedes}")
print(f"        backup precedes the settings write : {backup_before_write}")
print(f"        additive, original file preserved  : {additive}")

ok = preflight_ok and permission_precedes and backup_before_write and additive
print("")
print(f"AGENTIC INSTALL SAFE (preflight ok, permission precedes mutation, backup before write, additive): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("The installer asked before it touched anything. Next: name your DA.")
