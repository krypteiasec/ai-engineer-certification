#!/usr/bin/env python3
"""
LAB SDK4: Tool use (function calling).

A model on its own can only produce text. Tools let it ACT: check the weather,
run a query, call your function. You describe each tool with a name, a
description, and a JSON Schema for its inputs, then pass them on the request:

    tools = [{
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }]
    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024, tools=tools,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    )

When Claude decides to use a tool it does NOT run it. It stops with
stop_reason == "tool_use" and returns a tool_use block in `content`, carrying an
`id`, the tool `name`, and the `input` it chose. YOUR code executes the tool and
sends the answer back as a tool_result block (matched by tool_use_id). You can
steer the decision with tool_choice: "auto" (default), "any" (must use some
tool), {"type": "tool", "name": ...} (must use that one), or "none". In this lab
you define a tool registry and prove the model requests the RIGHT tool with a
well-formed tool_use block, which is the first half of the agent loop (next lab).

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk4-tool-use.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route


# --- Tool definitions, exactly the shape the Messages API expects. -----------
TOOLS = [
    {"name": "weather",
     "description": "Get the current weather for a city.",
     "input_schema": {"type": "object",
                      "properties": {"city": {"type": "string"}},
                      "required": ["city"]}},
    {"name": "calculator",
     "description": "Evaluate a simple arithmetic expression.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "search",
     "description": "Search the web for a query.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
]


def model_turn(user_text):
    """Simulate one model turn. The model reads the ask and either answers in text
    or asks for a tool. Here the offline provider's tool_route makes that choice;
    a real model decides internally and returns the same block shape."""
    chosen = tool_route(user_text, [t["name"] for t in TOOLS])
    if chosen is None:
        return {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": user_text}]}
    # A tool_use block: an id you will echo back, the tool name, and chosen input.
    tool_use = {"type": "tool_use", "id": "toolu_01ABC", "name": chosen,
                "input": {"city": "Paris"} if chosen == "weather" else {"query": user_text}}
    return {"stop_reason": "tool_use", "content": [tool_use]}


resp = model_turn("What's the weather in Paris right now?")

print("STEP 1: the model asked to use a tool")
print("  stop_reason :", resp["stop_reason"])
print("  content     :", resp["content"])

tool_use = next((b for b in resp["content"] if b["type"] == "tool_use"), None)

print("")
print("STEP 2: the tool_use block")
print("  id      :", tool_use["id"] if tool_use else None)
print("  name    :", tool_use["name"] if tool_use else None)
print("  input   :", tool_use["input"] if tool_use else None)

# --- You execute the tool and would return a tool_result (built in the next lab).
def run_weather(city):
    return "18C and clear in %s" % city

result_text = run_weather(tool_use["input"]["city"]) if tool_use else ""
tool_result = {"type": "tool_result", "tool_use_id": tool_use["id"], "content": result_text} \
    if tool_use else None

print("")
print("STEP 3: you run the tool and shape a tool_result")
print("  tool_result :", tool_result)

paused_for_tool = (resp["stop_reason"] == "tool_use")
right_tool = (tool_use is not None and tool_use["name"] == "weather")
block_well_formed = (tool_use is not None
                     and set(tool_use.keys()) >= {"type", "id", "name", "input"})
result_links_back = (tool_result is not None
                     and tool_result["tool_use_id"] == tool_use["id"])

print("")
print("  model stopped with stop_reason tool_use   :", paused_for_tool)
print("  it chose the correct tool (weather)        :", right_tool)
print("  tool_use block has id, name, input         :", block_well_formed)
print("  tool_result links back by tool_use_id      :", result_links_back)

ok = paused_for_tool and right_tool and block_well_formed and result_links_back
print("")
print("MODEL REQUESTS THE RIGHT TOOL VIA A TOOL_USE BLOCK: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("The model asks, you execute. Next: close the loop until the task is done.")
