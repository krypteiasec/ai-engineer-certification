#!/usr/bin/env python3
"""
LAB LO2: Install preflight and backup-before-overwrite.

The LifeOS installer never clobbers what you have. It verifies prerequisites
first (Bun, Git, the coding harness), and before it writes anything it backs up
your existing config to a timestamped copy. In this lab you simulate that flow on
a sample environment: check prerequisites, then prove the backup exists BEFORE
the new install is written. Everything is a fictional in-memory sample.

Run: python3 modules/academy-content/labs/lifeos/lo2-install-preflight.py
"""
import sys

# STEP 1: preflight. the installer confirms each required tool is present.
required = ["bun", "git", "harness"]
sample_env = {"bun": "1.1.0", "git": "2.44", "harness": "installed"}
missing = [t for t in required if t not in sample_env]
print("STEP 1: preflight, verify prerequisites")
for t in required:
    print(f"  {t:10} -> {'found ' + sample_env[t] if t in sample_env else 'MISSING'}")
preflight_ok = (missing == [])

# STEP 2: a sample existing install on disk, represented as a dict of files.
disk = {"config": "v-old", "identity": "old-name"}
events = []                                  # order of operations, for the invariant

def backup(store):
    snapshot = dict(store)                    # copy everything before touching it
    events.append("backup")
    return snapshot

def install(store):
    store["config"] = "v5.0.0"                # overwrite happens only after backup
    events.append("write")

print("")
print("STEP 2-3: back up, THEN write the new install")
backup_copy = backup(disk)
install(disk)
print(f"  operation order : {events}")
print(f"  backup preserved: config={backup_copy['config']!r} identity={backup_copy['identity']!r}")
print(f"  disk now        : config={disk['config']!r}")

# INVARIANT: preflight passed, backup ran strictly before the write, and the
# backup still holds the original bytes (nothing was lost).
ordered = events == ["backup", "write"]
preserved = backup_copy["config"] == "v-old" and backup_copy["identity"] == "old-name"
ok = preflight_ok and ordered and preserved
print("")
print(f"SAFE INSTALL (preflight ok, backup before write, originals preserved): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Prerequisites checked and your old config is safe. Next: capture your TELOS.")
