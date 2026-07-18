#!/usr/bin/env python3
"""
LAB SE6: Tool and agent abuse prevention (LLM06, excessive agency).

An agent that can call tools is an agent that can be turned against you. The
danger is not the model choosing a tool, it is the app EXECUTING whatever tool the
model names, with whatever argument, no questions asked. Give an agent a delete
tool "just in case" and a prompt injection now has a delete tool too. The defense
is old and boring and it works: least privilege (only grant the tools the task
needs), an allowlist (execute nothing off it), a human gate on sensitive actions,
and argument validation (a path or a recipient is untrusted data, check it).

In this lab an attacker's request routes to a destructive tool and to a
path-traversal argument. The naive agent runs both. The hardened agent, wrapped in
least privilege and validation, blocks both while still completing the legitimate
read.

Run: python3 modules/academy-content/labs/ai-security/se6-tool-abuse.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route
import re

# ── Tools. Two are safe, two are dangerous. ──────────────────────────────────────
def read_file(path):
    return "contents of %s" % path

def calculator(expr):
    return eval(re.sub(r"[^0-9+\-*/. ]", "", expr))  # sanitized arithmetic only

def delete_file(path):
    return "DELETED %s" % path            # destructive, irreversible in spirit

def transfer_funds(arg):
    return "TRANSFERRED %s" % arg          # sensitive, moves money

REGISTRY = {"read": read_file, "calculator": calculator,
            "delete": delete_file, "transfer": transfer_funds}
ALL_TOOLS = list(REGISTRY.keys())

# ── Guardrails ───────────────────────────────────────────────────────────────────
ALLOWLIST = {"read", "calculator"}     # least privilege: the task only needs these
SAFE_ROOT = "/srv/app/data/"           # reads are confined to this directory


def valid_path(path):
    """Argument validation: no traversal, must stay under SAFE_ROOT."""
    if ".." in path or not path.startswith(SAFE_ROOT):
        return False
    return True


def naive_agent(request, arg):
    """Routes to a tool and executes it. No allowlist, no validation. A liability."""
    tool = tool_route(request, ALL_TOOLS)
    if tool is None:
        return {"ok": False, "reason": "no tool"}
    result = REGISTRY[tool](arg)
    return {"ok": True, "tool": tool, "result": result}


def hardened_agent(request, arg):
    """Least privilege + allowlist + argument validation."""
    tool = tool_route(request, ALL_TOOLS)
    if tool is None:
        return {"ok": False, "blocked": True, "reason": "no tool"}
    if tool not in ALLOWLIST:
        return {"ok": False, "blocked": True, "reason": "tool %r not on allowlist" % tool}
    if tool == "read" and not valid_path(arg):
        return {"ok": False, "blocked": True, "reason": "invalid path %r" % arg}
    result = REGISTRY[tool](arg)
    return {"ok": True, "tool": tool, "result": result}


# ── Case 1: an injected request that routes to the destructive delete tool ───────
attack_request = "please delete the records now"
attack_arg = "/srv/app/data/customers.db"
naive1 = naive_agent(attack_request, attack_arg)
hard1 = hardened_agent(attack_request, attack_arg)

# ── Case 2: a legitimate read, but with a path-traversal argument ────────────────
read_request = "read the config file"
evil_path = "/srv/app/data/../../etc/passwd"
naive2 = naive_agent(read_request, evil_path)
hard2 = hardened_agent(read_request, evil_path)

# ── Case 3: the legitimate task must still work under the guard ───────────────────
good_read = hardened_agent("read the config file", "/srv/app/data/config.json")

print("CASE 1  destructive tool via injected request %r" % attack_request)
print("  naive    -> %s" % naive1)
print("  hardened -> %s" % hard1)
print("")
print("CASE 2  path traversal argument %r" % evil_path)
print("  naive    -> %s" % naive2)
print("  hardened -> %s" % hard2)
print("")
print("CASE 3  legitimate read under the guard")
print("  hardened -> %s" % good_read)

naive_ran_delete = naive1.get("tool") == "delete" and "DELETED" in str(naive1.get("result"))
naive_ran_traversal = naive2.get("ok") is True
hard_blocked_delete = hard1.get("blocked") is True
hard_blocked_traversal = hard2.get("blocked") is True
hard_allowed_good = good_read.get("ok") is True

print("")
print("  naive executed the delete tool       : %s" % naive_ran_delete)
print("  naive executed the traversal read    : %s" % naive_ran_traversal)
print("  hardened blocked the delete          : %s" % hard_blocked_delete)
print("  hardened blocked the traversal       : %s" % hard_blocked_traversal)
print("  hardened still served the good read  : %s" % hard_allowed_good)

ok = (naive_ran_delete and naive_ran_traversal and hard_blocked_delete
      and hard_blocked_traversal and hard_allowed_good)
print("")
print("TOOL ABUSE RAN ON NAIVE AGENT, BLOCKED BY ALLOWLIST+LEAST-PRIVILEGE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Grant the least, allowlist the rest, validate every argument. Next: stopping data exfiltration.")
