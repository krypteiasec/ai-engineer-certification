#!/usr/bin/env python3
"""
LAB CC6: Hooks and guardrails. Gate the agent before it acts.

An agent with tools and no limits is a liability. Hooks are how you keep one
inside its box: they run your code at fixed points in the agent lifecycle. The
one that matters most for safety is PreToolUse, which fires BEFORE any tool runs
and can allow, block, or ask for approval. On top of it you enforce two policies.
Least privilege: the agent may call only tools on an allowlist, everything else
is denied. Human in the loop: a sensitive or irreversible action is gated and
refused unless a human explicitly approves it. A second hook, PostToolUse, runs
after a tool and is where you log an audit trail. In this lab you wire a
PreToolUse gate and a PostToolUse audit log, then prove an allowed tool runs, a
non-allowlisted tool is blocked, and a destructive tool is refused without
approval but runs with it.

Reference (real Claude Agent SDK hook shape; the executed lab is offline):
    options=ClaudeAgentOptions(hooks={
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[my_gate])]})

Run: python3 modules/academy-content/labs/claude-code-agents/cc6-hooks.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

# ── The tools the agent could call. ─────────────────────────────────────────────
REGISTRY = {
    "read_file":  lambda arg: f"contents of {arg}",
    "calculator": lambda arg: sum(int(x) for x in arg.split("+")),
    "delete_all": lambda arg: f"DELETED {arg}",   # destructive, must be gated
    "send_email": lambda arg: f"emailed: {arg}",  # not granted at all
}

# ── Policy: least privilege (allowlist) + human-in-the-loop (sensitive). ─────────
ALLOWLIST = {"read_file", "calculator", "delete_all"}   # send_email is NOT granted
SENSITIVE = {"delete_all"}                               # needs explicit approval

audit = []   # PostToolUse fills this: the trail of what actually ran

def pre_tool_use(tool, approved):
    """PreToolUse HOOK: the gate. Returns a decision dict before any tool runs."""
    if tool not in ALLOWLIST:
        return {"decision": "deny", "reason": "not on allowlist"}
    if tool in SENSITIVE and not approved:
        return {"decision": "ask", "reason": "sensitive action needs approval"}
    return {"decision": "allow"}

def post_tool_use(tool, result):
    """PostToolUse HOOK: append to the audit trail."""
    audit.append((tool, result))

def call(tool, arg, approved=False):
    """Every tool call passes through PreToolUse; only allowed ones run + log."""
    decision = pre_tool_use(tool, approved)
    if decision["decision"] != "allow":
        return {"ran": False, **decision}
    result = REGISTRY[tool](arg)
    post_tool_use(tool, result)
    return {"ran": True, "decision": "allow", "result": result}

# ── 1. Allowed tool runs. ───────────────────────────────────────────────────────
r_allow = call("calculator", "2+2")
print(f"calculator('2+2')              -> {r_allow}")

# ── 2. Non-allowlisted tool is blocked. ─────────────────────────────────────────
r_block = call("send_email", "leak the data")
print(f"send_email(...)  [not granted] -> {r_block}")

# ── 3. Sensitive tool is gated without approval, runs with it. ──────────────────
r_gate = call("delete_all", "/tmp/x")                 # no approval
print(f"delete_all(...)  [no approval] -> {r_gate}")
r_ok = call("delete_all", "/tmp/x", approved=True)    # human approved
print(f"delete_all(...)  [approved]    -> {r_ok}")

print("")
print(f"audit trail (PostToolUse): {audit}")

allowed_ran = r_allow["ran"] and r_allow["result"] == 4
blocked = r_block["ran"] is False and r_block["decision"] == "deny"
gated = r_gate["ran"] is False and r_gate["decision"] == "ask"
approved_ran = r_ok["ran"] and r_ok["result"] == "DELETED /tmp/x"
audit_clean = audit == [("calculator", 4), ("delete_all", "DELETED /tmp/x")]

print("")
print(f"allowed tool ran                    : {allowed_ran}")
print(f"ungranted tool blocked              : {blocked}")
print(f"sensitive tool gated without approval: {gated}")
print(f"sensitive tool ran once approved    : {approved_ran}")
print(f"audit logged only what ran          : {audit_clean}")

ok = allowed_ran and blocked and gated and approved_ran and audit_clean
print("")
print(f"PRETOOLUSE HOOK ENFORCED THE GUARDRAILS: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Gate before acting, log after. Next: orchestrate a whole fleet of agents.")
