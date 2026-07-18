#!/usr/bin/env python3
"""
LAB AM6: The Model Context Protocol (MCP).

Every agent framework used to invent its own way to expose tools, so nothing was
reusable. MCP is Anthropic's open standard that fixes this: one protocol, so a
tool you build once plugs into any MCP-aware app. It speaks JSON-RPC, a tiny
request/response format: the client sends {method, params, id}, the server
answers {result, id}. The three calls that matter here are `initialize` (the
handshake), `tools/list` (what can I call?), and `tools/call` (call it). In this
lab you implement a real, if small, MCP server in memory and drive it as a
client, and prove a tools/call returns the correct answer through the protocol.

Run: python3 modules/academy-content/labs/agents-mcp/am6-mcp-protocol.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
import re


def _calculator(expression):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expression)
    if not m:
        raise ValueError("cannot parse %r" % expression)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r


class MCPServer:
    """A minimal MCP server. It exposes tools and answers JSON-RPC requests.
    This is genuinely how a real MCP server is shaped, just without a network."""

    def __init__(self, name):
        self.name = name
        self._tools = {
            "calculator": {
                "schema": {"name": "calculator",
                           "description": "Evaluate an arithmetic expression.",
                           "parameters": {"expression": {"type": "string"}}},
                "fn": _calculator,
            },
        }

    def handle(self, request):
        """The JSON-RPC entry point: one request in, one response out."""
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05",
                          "serverInfo": {"name": self.name},
                          "capabilities": {"tools": {}}}
            elif method == "tools/list":
                result = {"tools": [t["schema"] for t in self._tools.values()]}
            elif method == "tools/call":
                tool = self._tools[params["name"]]
                value = tool["fn"](**params["arguments"])
                # MCP wraps tool output as a list of content blocks.
                result = {"content": [{"type": "text", "text": str(value)}]}
            else:
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601, "method": method}}
            return {"jsonrpc": "2.0", "id": rid, "result": result}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32000, "message": str(e)}}


def call(server, _id, method, **params):
    """Client side: build a JSON-RPC request and hand it to the server."""
    return server.handle({"jsonrpc": "2.0", "id": _id, "method": method, "params": params})


server = MCPServer("academy-calc")

# 1. HANDSHAKE. Every MCP session opens with initialize.
init = call(server, 1, "initialize")
print("STEP 1: initialize ->", init["result"]["serverInfo"])

# 2. DISCOVERY. Ask the server what tools it offers.
listed = call(server, 2, "tools/list")
tool_names = [t["name"] for t in listed["result"]["tools"]]
print("STEP 2: tools/list ->", tool_names)

# 3. INVOCATION. Call a tool through the protocol and read the result block.
called = call(server, 3, "tools/call", name="calculator", arguments={"expression": "144 / 12"})
text = called["result"]["content"][0]["text"]
print("STEP 3: tools/call -> content:", text)

handshake_ok = init["result"]["protocolVersion"] == "2024-11-05"
discovery_ok = "calculator" in tool_names
call_ok = (text == "12")
# The protocol must also report errors, not crash, on a bad call.
err = call(server, 4, "tools/call", name="calculator", arguments={"expression": "nope"})
error_ok = "error" in err

print("")
print(f"handshake returned a protocol version : {handshake_ok}")
print(f"tools/list advertised the calculator  : {discovery_ok}")
print(f"tools/call returned the right result   : {call_ok}")
print(f"a bad call returned a JSON-RPC error   : {error_ok}")

ok = handshake_ok and discovery_ok and call_ok and error_ok
print("")
print(f"MCP SERVER ANSWERED TOOLS/CALL: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("One protocol, any client. Next: run it over a real transport with resources.")
