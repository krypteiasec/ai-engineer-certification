#!/usr/bin/env python3
"""
CAPSTONE 5 (ch5): An MCP-connected assistant (server + client + a tool).

Capstone project four: an assistant that reaches its capabilities through the
Model Context Protocol instead of hard-wiring them. This is the Agents-course MCP
lab grown into a small end-to-end assistant. There are three real pieces: an MCP
SERVER that exposes a tool (a calculator) and a resource (an owner note) behind a
JSON transport, an MCP CLIENT that only ever exchanges JSON strings across that
transport (no shared memory), and an ASSISTANT that, when a user asks a math
question, discovers the tool over MCP, calls it, and answers from the result.

The point of MCP is build-once-connect-everywhere: the assistant did not import a
calculator, it discovered and called one over a standard protocol, so any
MCP-aware host could use the same server. The lab drives a real user question
through the whole assistant -> client -> transport -> server -> tool path and
proves the served tool call produced the correct answer, then prints the
invariant and exits 0.

Run: python3 modules/academy-content/labs/capstones/cap5-mcp-assistant.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
import json, re


def _calculator(expression):
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)", expression)
    if not m:
        raise ValueError("cannot parse %r" % expression)
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    r = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "/": a / b if b else float("nan")}[op]
    return int(r) if r == int(r) else r


class MCPServer:
    """Exposes a TOOL and a RESOURCE. It sees only a JSON string and returns a
    JSON string, exactly like the stdio transport, minus the real pipe."""

    TOOLS = {"calculator": {"description": "Evaluate arithmetic.", "fn": _calculator}}
    RESOURCES = {"notes://owner": "This portfolio assistant belongs to Ada."}

    def transport(self, line):
        try:
            req = json.loads(line)
        except Exception:
            return json.dumps({"jsonrpc": "2.0", "id": None,
                               "error": {"code": -32700, "message": "parse error"}})
        rid, method, params = req.get("id"), req.get("method"), req.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05",
                          "capabilities": {"tools": {}, "resources": {}}}
            elif method == "tools/list":
                result = {"tools": [{"name": n, "description": t["description"]}
                                    for n, t in self.TOOLS.items()]}
            elif method == "tools/call":
                value = self.TOOLS[params["name"]]["fn"](**params["arguments"])
                result = {"content": [{"type": "text", "text": str(value)}]}
            elif method == "resources/read":
                result = {"contents": [{"uri": params["uri"],
                                        "text": self.RESOURCES[params["uri"]]}]}
            else:
                return json.dumps({"jsonrpc": "2.0", "id": rid,
                                   "error": {"code": -32601, "message": method}})
            return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "id": rid,
                               "error": {"code": -32000, "message": str(e)}})


class MCPClient:
    """Knows the server ONLY through the transport function. Everything on the
    wire is a JSON string, never a Python object."""

    def __init__(self, transport):
        self._transport, self._id = transport, 0

    def request(self, method, **params):
        self._id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        return json.loads(self._transport(line))


class MCPAssistant:
    """The user-facing assistant. It answers by DISCOVERING and CALLING tools over
    MCP, not by importing them, so any MCP-aware host could reuse the server."""

    def __init__(self, client):
        self.client = client
        self.client.request("initialize")
        # Discover what the server offers, over the wire.
        self.tools = [t["name"] for t in
                      self.client.request("tools/list")["result"]["tools"]]

    def answer(self, question):
        m = re.search(r"(-?\d+\s*[+\-*/x]\s*-?\d+)", question)
        if m and "calculator" in self.tools:
            call = self.client.request("tools/call", name="calculator",
                                       arguments={"expression": m.group(1)})
            value = call["result"]["content"][0]["text"]
            return "The answer is %s." % value, value
        return "I cannot answer that with my tools.", None


def main():
    server = MCPServer()
    client = MCPClient(server.transport)   # client holds only the transport
    assistant = MCPAssistant(client)
    print("STEP 1: assistant discovered tools over MCP -> %s\n" % assistant.tools)

    question = "What is 24 * 7 for my weekly total?"
    print("STEP 2: user asks -> %r" % question)
    reply, value = assistant.answer(question)
    print("  assistant -> %r  (tool result over the wire: %s)" % (reply, value))

    # Confirm the whole path worked: the tool was discovered, called over the
    # transport, and returned the correct arithmetic (24 * 7 = 168).
    tool_discovered = "calculator" in assistant.tools
    served_value_correct = (value == "168")
    answer_carries_it = "168" in reply

    print("")
    print("  tool discovered over MCP: %s" % ("YES" if tool_discovered else "NO"))
    print("  tool call served correct result over transport: %s" %
          ("YES" if served_value_correct else "NO"))

    ok = tool_discovered and served_value_correct and answer_carries_it
    print("")
    print("MCP ASSISTANT SERVED A TOOL CALL: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Server + client + tool, over a standard protocol. Build once, connect everywhere. Capstone four.")


if __name__ == "__main__":
    main()
