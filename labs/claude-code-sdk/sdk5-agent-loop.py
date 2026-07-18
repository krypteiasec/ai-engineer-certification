#!/usr/bin/env python3
"""
LAB SDK5: The agent loop (the tool_runner concept).

One tool call is not an agent. An AGENT is the LOOP: the model asks for a tool,
you run it, you feed the result back, and the model decides what to do next, over
and over, until it stops asking for tools and gives a final answer. In code the
loop is:

    while resp.stop_reason == "tool_use":
        for block in resp.content:
            if block.type == "tool_use":
                out = run_tool(block.name, block.input)     # you execute
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": out}
                ]})
        resp = client.messages.create(model="claude-opus-4-8",
                                      max_tokens=1024, tools=tools, messages=messages)
    # loop exits when stop_reason == "end_turn": that is the final answer.

The SDK will drive this loop for you with a TOOL RUNNER helper
(client.beta.messages.tool_runner in Python, .toolRunner in TypeScript): you just
supply the tool functions and it handles request, execute, feed-back, repeat.
Writing the loop by hand, as you do here, is the way to understand what the tool
runner automates.

Note the naming trap. The API Tool Runner above is NOT the Claude Agent SDK. The
Claude Agent SDK (claude-agent-sdk) is Claude Code packaged as a library, with
built-in Read, Write, Edit, Bash, and web tools and its own harness. The Tool
Runner loops over tools YOU define. Same idea, different scope. This lab builds
the plain loop that sits under both.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk5-agent-loop.py
"""
import sys, os, re
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route


# --- The one tool the agent can call. ----------------------------------------
def calculator(expression):
    """Evaluate 'A plus B'. Real tools do real work; this one is deterministic."""
    m = re.search(r"(-?\d+)\s*(?:plus|\+)\s*(-?\d+)", expression.lower())
    if m:
        return str(int(m.group(1)) + int(m.group(2)))
    return "error"


TOOLS = ["calculator", "weather", "search"]


def model_turn(messages):
    """Simulate the model. If the last message is a tool_result, the model has
    what it needs and returns a FINAL text answer (end_turn). Otherwise it reads
    the user ask and, if a tool fits, requests it (tool_use)."""
    last = messages[-1]
    # A tool_result came back: produce the final answer and stop.
    if last["role"] == "user" and isinstance(last["content"], list) \
            and last["content"] and last["content"][0].get("type") == "tool_result":
        answer = last["content"][0]["content"]
        return {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "The answer is %s." % answer}]}
    # Otherwise: read the ask, maybe request a tool.
    user_text = last["content"] if isinstance(last["content"], str) else ""
    chosen = tool_route(user_text, TOOLS)
    if chosen == "calculator":
        return {"stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "toolu_01",
                             "name": "calculator", "input": {"expression": user_text}}]}
    return {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": "I can answer that directly."}]}


# --- The agent loop. ---------------------------------------------------------
messages = [{"role": "user", "content": "Please calculate 2 plus 2 with the calculator."}]
resp = model_turn(messages)

iterations, tool_calls, MAX = 0, 0, 8
print("STEP 1: run the loop until the model stops asking for tools")
while resp["stop_reason"] == "tool_use" and iterations < MAX:
    iterations += 1
    for block in resp["content"]:
        if block["type"] == "tool_use":
            out = calculator(block["input"]["expression"])
            tool_calls += 1
            print("  round %d: model called %s(%r) -> %r"
                  % (iterations, block["name"], block["input"]["expression"], out))
            messages.append({"role": "assistant", "content": resp["content"]})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block["id"], "content": out}]})
    resp = model_turn(messages)

final_text = "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")

print("")
print("STEP 2: the loop finished")
print("  iterations   :", iterations)
print("  tool calls   :", tool_calls)
print("  stop_reason  :", resp["stop_reason"])
print("  final answer :", repr(final_text))

looped = (iterations >= 1)
used_tool = (tool_calls >= 1)
ended_clean = (resp["stop_reason"] == "end_turn")
answer_ok = ("4" in final_text)
bounded = (iterations < MAX)   # the MAX guard prevents an infinite loop

print("")
print("  the loop ran at least one tool round       :", looped)
print("  a tool was actually executed               :", used_tool)
print("  the loop exited on end_turn                 :", ended_clean)
print("  the final answer is correct (4)             :", answer_ok)
print("  a max-iteration guard bounded the loop      :", bounded)

ok = looped and used_tool and ended_clean and answer_ok and bounded
print("")
print("THE AGENT LOOP RAN TOOLS UNTIL END_TURN: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Ask, execute, feed back, repeat. Next: make the output a schema you can trust.")
