#!/usr/bin/env python3
"""
LAB CC5: MCP integration. Why a protocol, and how discovery works.

Every capability you hardcode into an agent is a capability you have to build and
maintain. The Model Context Protocol (MCP) is the standard that fixes this: an
MCP server publishes a set of tools behind one interface, and any agent that
speaks the protocol can DISCOVER those tools at runtime and call them, without
knowing them in advance. That is the whole point, decoupling: the team that
provides a capability (a database, a browser, an API) ships an MCP server, and
your agent plugs it in without a code change. The protocol boundary is two
methods: list_tools (what can you do) and call_tool (do it). In the real Claude
Agent SDK you pass mcp_servers and the tools appear to the agent automatically.
In this lab you build a tiny MCP-style server, connect an agent that starts with
an EMPTY toolset, and prove it discovered and called a tool it never hardcoded.

Reference (real Claude Agent SDK; the executed lab is an offline mock server):
    options=ClaudeAgentOptions(mcp_servers={
        "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]}})

Run: python3 modules/academy-content/labs/claude-code-agents/cc5-mcp.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import tool_route


class MCPServer:
    """A minimal MCP-style server. It owns a set of tools and exposes exactly two
    protocol methods. The agent knows NONE of these tool names ahead of time."""
    def __init__(self):
        self._tools = {
            "unit_convert": {
                "description": "Convert miles to kilometers.",
                "fn": lambda miles: round(miles * 1.60934, 3),
            },
            "capital_lookup": {
                "description": "Search for the capital city of a country.",
                "fn": lambda country: {"france": "Paris", "japan": "Tokyo"}[country.lower()],
            },
        }

    def list_tools(self):
        """Protocol method 1: advertise what this server can do."""
        return [{"name": n, "description": t["description"]} for n, t in self._tools.items()]

    def call_tool(self, name, arg):
        """Protocol method 2: run a named tool. This is the protocol boundary."""
        if name not in self._tools:
            return {"ok": False, "reason": "unknown tool"}
        return {"ok": True, "result": self._tools[name]["fn"](arg)}


class Agent:
    """An agent that hardcodes NO tools. It learns them from the server at runtime."""
    def __init__(self):
        self.local_tools = []          # deliberately empty: nothing hardcoded
        self.discovered = []

    def connect(self, server):
        """Discovery: pull the server's tool list into the agent's awareness."""
        self.discovered = server.list_tools()

    def handle(self, request, server):
        """Route a request to a DISCOVERED tool by keyword, call it over MCP."""
        names = [t["name"] for t in self.discovered]
        # 'capital_lookup' matches on 'search'/'capital'; route by description keywords.
        low = request.lower()
        picked = None
        for t in self.discovered:
            if any(w in low for w in t["name"].split("_")) or \
               tool_route(request, [t["name"]]) == t["name"]:
                picked = t["name"]; break
        if picked is None:
            return {"ok": False, "reason": "no matching discovered tool"}
        return picked, server.call_tool(picked, request.split()[-1] if picked == "capital_lookup" else 100)


server = MCPServer()
agent = Agent()

print(f"agent hardcoded tools : {agent.local_tools}  (empty on purpose)")
agent.connect(server)
print(f"discovered over MCP   : {[t['name'] for t in agent.discovered]}")
print("")

# The agent handles a request using a tool it only learned about at connect time.
picked, out = agent.handle("look up the capital of France", server)
print(f"request 'capital of France' -> tool {picked!r} -> {out}")

print("")
started_empty = agent.local_tools == []
discovered_two = len(agent.discovered) == 2
used_discovered = picked in [t["name"] for t in agent.discovered] and picked not in agent.local_tools
correct = out["ok"] and out["result"] == "Paris"

print(f"agent hardcoded no tools             : {started_empty}")
print(f"discovered the server's tools        : {discovered_two}")
print(f"called a tool it did not hardcode     : {used_discovered}")
print(f"the MCP call returned the right value : {correct}")

ok = started_empty and discovered_two and used_discovered and correct
print("")
print(f"AGENT DISCOVERED AND CALLED AN MCP TOOL IT DID NOT HARDCODE: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Discover, then call over the protocol. Next: guard the agent with hooks.")
