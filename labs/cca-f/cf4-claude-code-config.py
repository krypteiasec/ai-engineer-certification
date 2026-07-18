#!/usr/bin/env python3
"""
LAB CF4: Claude Code configuration and workflows (Domain 3, 20%).

Where a rule lives decides who gets it. This lab routes configuration items to
the correct scope and file, and reproduces the single most common exam trap: a
new teammate never receives the project coding standards because they were put
in user-level ~/.claude/CLAUDE.md (not shared) instead of project-level
.claude/CLAUDE.md (shared via version control). It also matches path-scoped
rule files to the files they should load for, and picks planning mode vs direct
execution.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf4-claude-code-config.py
"""
import sys
import fnmatch

# Each config file: (scope, shared via VCS?). Straight from the study guide.
FILES = {
    "~/.claude/CLAUDE.md":        ("user, all projects",        False),
    ".claude/CLAUDE.md":          ("project, all contributors", True),
    "subdir/CLAUDE.md":           ("that directory and below",  True),
    "~/.claude/settings.json":    ("machine-wide",              False),
    ".claude/settings.json":      ("project-wide permissions",  True),
    ".claude/settings.local.json":("personal project override", False),
    ".mcp.json":                  ("project MCP servers",       True),
    "~/.claude.json":             ("personal MCP servers",      False),
}


def is_shared(path: str) -> bool:
    return FILES[path][1]

# STEP 1: route each requirement to the correct file.
# (requirement, correct file)
ROUTING = [
    ("Team coding standards everyone must follow", ".claude/CLAUDE.md"),
    ("A personal preference for just my machine",  "~/.claude/CLAUDE.md"),
    ("Tool permission allowlist for the whole team", ".claude/settings.json"),
    ("A shared MCP server for all contributors",   ".mcp.json"),
    ("My own experimental MCP server",             "~/.claude.json"),
]
print("STEP 1: route configuration to the correct file")
routing_ok = True
for req, path in ROUTING:
    shared = is_shared(path)
    print(f"  {req}\n      -> {path}  (shared={shared})")
    # team-wide things MUST be shared; personal things MUST NOT be
    expect_shared = req.lower().startswith(("team", "tool permission", "a shared"))
    routing_ok = routing_ok and (shared == expect_shared)

# STEP 2: the classic trap. Standards placed user-level do not reach a teammate.
print("")
print("STEP 2: the 'new teammate misses standards' trap")
wrong_placement = "~/.claude/CLAUDE.md"
right_placement = ".claude/CLAUDE.md"
teammate_gets_wrong = is_shared(wrong_placement)
teammate_gets_right = is_shared(right_placement)
print(f"  standards in {wrong_placement:<22} reach teammate? {teammate_gets_wrong}")
print(f"  standards in {right_placement:<22} reach teammate? {teammate_gets_right}")

# STEP 3: path-scoped rules load only for matching files.
RULE_PATHS = ["**/*.test.ts", "**/*.test.tsx"]


def rule_loads_for(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in RULE_PATHS)

print("")
print("STEP 3: .claude/rules/ path matching")
cases = [("src/auth/login.test.ts", True), ("src/auth/login.ts", False)]
paths_ok = True
for fname, expect in cases:
    got = rule_loads_for(fname)
    paths_ok = paths_ok and (got == expect)
    print(f"  {fname:<26} loads test rule? {got}")

# STEP 4: planning mode vs direct execution.
def mode_for(task: str) -> str:
    big = task in ("cross-codebase refactor", "unfamiliar codebase",
                   "architectural decision")
    return "planning" if big else "direct"

print("")
print("STEP 4: workflow mode")
mode_cases = [("cross-codebase refactor", "planning"),
              ("single-file fix with clear stack trace", "direct")]
mode_ok = True
for task, expect in mode_cases:
    got = mode_for(task)
    mode_ok = mode_ok and (got == expect)
    print(f"  {task:<40} -> {got} mode")

# Invariants.
trap_ok = (teammate_gets_wrong is False and teammate_gets_right is True)
ok = routing_ok and trap_ok and paths_ok and mode_ok
print("")
print(f"  config routed to correct scope        : {routing_ok}")
print(f"  teammate-misses-standards trap proven : {trap_ok}")
print(f"  path-scoped rules match correctly     : {paths_ok}")
print(f"  planning vs direct mode chosen right  : {mode_ok}")
print("")
print(f"CONFIG ROUTED TO CORRECT SCOPE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Shared standards go project-level and get committed. settings.json holds permissions. Next: prompt engineering.")
