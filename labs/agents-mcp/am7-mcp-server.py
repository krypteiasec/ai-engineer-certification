#!/usr/bin/env python3
"""
LAB AM7: Building an MCP server over a real transport.

Lab AM6 called the server's method directly. A real MCP server never shares
memory with its client. They are separate processes that talk over a TRANSPORT:
the client serializes each request to a JSON string, writes it as a line, and
reads a JSON line back. The stdio transport (newline-delimited JSON) is exactly
this. And a server exposes more than tools: MCP has three primitives, TOOLS
(actions the model can call), RESOURCES (data it can read), and PROMPTS (reusable
templates). In this lab you build a server that offers all three, then drive it
through a client that only ever exchanges JSON STRINGS, proving the whole thing
works across the transport boundary with no shared state.

Run: python3 modules/academy-content/labs/agents-mcp/am7-mcp-server.py
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
    """A server exposing all three MCP primitives. It only ever sees a JSON
    STRING (a transport line) and returns a JSON STRING. No shared objects."""

    TOOLS = {"calculator": {"description": "Evaluate arithmetic.", "fn": _calculator}}
    RESOURCES = {
        "notes://mission": "The mission is to teach AI engineering hands on.",
        "notes://owner": "This academy belongs to Ada.",
    }
    PROMPTS = {"summarize": "Summarize the following text in one sentence: {text}"}

    def transport(self, line):
        """One newline-delimited JSON request line in, one response line out.
        This IS the stdio transport, minus the actual pipe."""
        try:
            req = json.loads(line)
        except Exception:
            return json.dumps({"jsonrpc": "2.0", "id": None,
                               "error": {"code": -32700, "message": "parse error"}})
        rid, method, params = req.get("id"), req.get("method"), req.get("params", {})
        try:
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05",
                          "capabilities": {"tools": {}, "resources": {}, "prompts": {}}}
            elif method == "tools/list":
                result = {"tools": [{"name": n, "description": t["description"]}
                                    for n, t in self.TOOLS.items()]}
            elif method == "tools/call":
                value = self.TOOLS[params["name"]]["fn"](**params["arguments"])
                result = {"content": [{"type": "text", "text": str(value)}]}
            elif method == "resources/list":
                result = {"resources": [{"uri": u} for u in self.RESOURCES]}
            elif method == "resources/read":
                text = self.RESOURCES[params["uri"]]
                result = {"contents": [{"uri": params["uri"], "text": text}]}
            elif method == "prompts/get":
                tmpl = self.PROMPTS[params["name"]]
                result = {"messages": [{"role": "user",
                                        "content": tmpl.format(**params.get("arguments", {}))}]}
            else:
                return json.dumps({"jsonrpc": "2.0", "id": rid,
                                   "error": {"code": -32601, "message": method}})
            return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as e:
            return json.dumps({"jsonrpc": "2.0", "id": rid,
                               "error": {"code": -32000, "message": str(e)}})


class MCPClient:
    """The client. It knows the server ONLY through the transport function, and
    everything it sends or receives is a JSON string, never a Python object."""

    def __init__(self, transport):
        self._transport = transport
        self._id = 0

    def request(self, method, **params):
        self._id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        response_line = self._transport(line)     # <-- the wire: string out, string in
        return json.loads(response_line)


server = MCPServer()
client = MCPClient(server.transport)   # the client holds only the transport

# 1. Initialize across the wire.
init = client.request("initialize")
caps = sorted(init["result"]["capabilities"].keys())
print("STEP 1: initialize -> capabilities:", caps)

# 2. List and CALL a tool over the transport.
tools = [t["name"] for t in client.request("tools/list")["result"]["tools"]]
called = client.request("tools/call", name="calculator", arguments={"expression": "9 * 9"})
tool_text = called["result"]["content"][0]["text"]
print("STEP 2: tools ->", tools, "| tools/call 9*9 ->", tool_text)

# 3. Read a RESOURCE over the transport (data, not an action).
res = client.request("resources/read", uri="notes://mission")
res_text = res["result"]["contents"][0]["text"]
print("STEP 3: resources/read notes://mission ->", repr(res_text))

# 4. Get a PROMPT template filled in over the transport.
pr = client.request("prompts/get", name="summarize", arguments={"text": "hello world"})
prompt_text = pr["result"]["messages"][0]["content"]
print("STEP 4: prompts/get summarize ->", repr(prompt_text))

caps_ok = caps == ["prompts", "resources", "tools"]
tool_ok = (tool_text == "81")
resource_ok = "mission" in res_text
prompt_ok = "hello world" in prompt_text

print("")
print(f"all three primitives advertised   : {caps_ok}")
print(f"tools/call worked over transport  : {tool_ok}")
print(f"resources/read worked over wire   : {resource_ok}")
print(f"prompts/get filled the template   : {prompt_ok}")

ok = caps_ok and tool_ok and resource_ok and prompt_ok
print("")
print(f"MCP CLIENT CALLED THE SERVER OVER THE TRANSPORT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Tools, resources, prompts, over the wire. Next: put guardrails on the agent.")
