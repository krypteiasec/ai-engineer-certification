#!/usr/bin/env python3
"""
LAB SDK1: The Messages API. One endpoint, one shape.

Everything you do with Claude goes through a single endpoint, POST /v1/messages,
and a single call in the SDK:

    from anthropic import Anthropic
    client = Anthropic()                       # reads ANTHROPIC_API_KEY, or a saved profile
    resp = client.messages.create(
        model="claude-opus-4-8",               # the current Opus model id
        max_tokens=1024,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    print(resp.content[0].text)                # -> "Paris"

The response is not a bare string. It is a structured object: a `content` list of
typed blocks (a text block has type "text" and a `text` field), a `usage` object
with `input_tokens` and `output_tokens`, and a `stop_reason` that tells you WHY
generation stopped ("end_turn" is a clean finish). Learn this shape once and it
transfers to every Claude call you will ever write. In this lab you build a tiny
client with the exact Messages API shape over the shared offline provider and
prove the response carries typed content blocks, usage, and a stop reason.

Real model ids you would pass to `model=` today: claude-opus-4-8 (most capable
Opus), claude-sonnet-5 (balanced), claude-haiku-4-5 (fast and cheap). This lab
runs fully offline against academy_llm, so no key and no network are needed.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk1-messages-api.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat


def count_tokens(text):
    # A token here is a whitespace word. Real SDKs count subword tokens (see the
    # count_tokens endpoint in lab SDK7), but a word count teaches usage cleanly.
    return len(text.split())


class Message:
    """Mirrors what client.messages.create returns. `content` is a LIST of typed
    blocks (not a string), `usage` accounts tokens, `stop_reason` says why the
    model stopped. Downstream code reads `.text` without caring about internals."""
    def __init__(self, model, text, in_tokens, out_tokens, stop_reason="end_turn"):
        self.model = model
        self.content = [{"type": "text", "text": text}]
        self.usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}
        self.stop_reason = stop_reason

    @property
    def text(self):
        return "".join(b["text"] for b in self.content if b["type"] == "text")


class Anthropic:
    """A minimal stand-in for the real `anthropic.Anthropic` client. The one call
    you use most is messages.create(model, max_tokens, system, messages). `system`
    is a SEPARATE top-level field, not a message in the list. That distinction is
    the subject of the next lab."""
    def __init__(self, model_default="claude-opus-4-8"):
        self.model_default = model_default

    def messages_create(self, messages, model=None, max_tokens=1024, system=None):
        model = model or self.model_default
        convo = []
        if system:
            convo.append({"role": "system", "content": system})
        convo.extend(messages)
        reply = chat(convo)                      # the offline provider does the work
        # A real API would cap output at max_tokens and set stop_reason to
        # "max_tokens" if it hit the cap. Our reply is short, so it ends cleanly.
        stop = "max_tokens" if count_tokens(reply) >= max_tokens else "end_turn"
        in_text = " ".join(m.get("content", "") for m in convo)
        return Message(model, reply, count_tokens(in_text), count_tokens(reply), stop)


# --- Call it exactly like the real SDK. --------------------------------------
client = Anthropic()
resp = client.messages_create(
    model="claude-opus-4-8",
    max_tokens=1024,
    system="You are a classifier. You label sentiment.",
    messages=[{"role": "user",
               "content": "Classify the sentiment. Input: I love this product, it works perfectly"}],
)

print("STEP 1: the response is a structured object, not a bare string")
print("  model        :", resp.model)
print("  content      :", resp.content)
print("  usage        :", resp.usage)
print("  stop_reason  :", resp.stop_reason)
print("  .text        :", repr(resp.text))

# --- Verify the Messages API shape your app relies on. -----------------------
shape_ok = (
    isinstance(resp.content, list)
    and resp.content and resp.content[0].get("type") == "text"
    and set(resp.usage.keys()) == {"input_tokens", "output_tokens"}
    and resp.stop_reason == "end_turn"
)
model_ok = (resp.model == "claude-opus-4-8")
answer_ok = (resp.text == "positive")
usage_ok = (resp.usage["input_tokens"] > 0 and resp.usage["output_tokens"] > 0)

print("")
print("STEP 2: checks")
print("  content is a list of typed blocks        :", shape_ok)
print("  model id was carried on the response      :", model_ok)
print("  the model answered correctly              :", answer_ok)
print("  token usage was accounted                 :", usage_ok)

ok = shape_ok and model_ok and answer_ok and usage_ok
print("")
print("MESSAGES API RETURNS TYPED CONTENT BLOCKS: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("One endpoint, one shape. Next: system prompts and the role hierarchy.")
