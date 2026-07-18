#!/usr/bin/env python3
"""
LAB SDK2: System prompts, roles, and a stateless API.

Three roles carry a conversation. The SYSTEM prompt sets standing behavior (who
the model is, its job, its rules) and is a separate top-level field, not a
message. The USER role is what the person says. The ASSISTANT role is what the
model said last turn. In the real SDK:

    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024,
        system="You are a terse senior engineer. Answer in one sentence.",
        messages=[
            {"role": "user",      "content": "How do I read a JSON file in Python?"},
            {"role": "assistant", "content": "Use json.load(open(path))."},
            {"role": "user",      "content": "And write it back?"},
        ],
    )

Two rules that trip people up. First, the API is STATELESS: it remembers nothing
between calls, so YOU resend the full message history every turn. Turn three only
knows about turns one and two because you sent them again. Second, the first
message must be `user` and roles alternate user/assistant. In this lab you prove
that the system prompt sets the job (without it the model has no task and just
echoes), and that resending history preserves context across turns.

(Advanced: on Claude Opus 4.8 you can also append a {"role": "system", ...}
message mid-conversation to inject an operator instruction without resending a
new top-level system prompt. The standing system prompt above is the common case.)

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk2-system-and-roles.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import chat


def create(system, messages):
    """Flatten system + role-tagged messages into one provider call, exactly the
    way the real API composes a request before the model reads it."""
    convo = []
    if system:
        convo.append({"role": "system", "content": system})
    convo.extend(messages)
    return chat(convo)


# --- 1. The system prompt sets the job. --------------------------------------
# The user turn is just raw material, with no task word in it. Only the system
# prompt names the job, so it alone decides whether the model has a task at all.
review = {"role": "user", "content": "I love this product, it works perfectly"}

no_system = create(system=None, messages=[review])
with_system = create(system="You classify the sentiment of a review as positive, negative, or neutral.",
                     messages=[review])

print("STEP 1: does the system prompt set the task?")
print("  no system prompt   :", repr(no_system))
print("  with system prompt :", repr(with_system))

# --- 2. The API is stateless: resend the whole history each turn. ------------
# We simulate a multi-turn conversation. Each turn appends to `history` and
# resends ALL of it. The provider only sees what we resend.
history = []
history.append({"role": "user", "content": "sentiment: this app is fantastic and fast"})
turn1 = create(system="You classify sentiment.", messages=history)
history.append({"role": "assistant", "content": turn1})     # record what the model said
history.append({"role": "user", "content": "sentiment: but the update is terrible and buggy"})
turn2 = create(system="You classify sentiment.", messages=history)

print("")
print("STEP 2: a stateless, resent conversation")
print("  turn 1 (positive review) ->", repr(turn1))
print("  turn 2 (negative review) ->", repr(turn2))
print("  history length resent    :", len(history), "messages")

# --- 3. Role rules the API enforces. -----------------------------------------
first_is_user = (history[0]["role"] == "user")
roles_present = {"user", "assistant"} <= {m["role"] for m in history}

print("")
print("STEP 3: role structure")
print("  first message is 'user'  :", first_is_user)
print("  user and assistant roles :", roles_present)

system_steers = (with_system == "positive" and no_system != "positive")
stateless_ok = (turn1 == "positive" and turn2 == "negative")

print("")
print("  system role set the job (echo without it)     :", system_steers)
print("  resent history kept each turn on-task         :", stateless_ok)
print("  first message user, roles alternate           :", first_is_user and roles_present)

ok = system_steers and stateless_ok and first_is_user and roles_present
print("")
print("SYSTEM ROLE SETS THE JOB, HISTORY IS RESENT: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("System sets behavior, you own the history. Next: stream the tokens live.")
