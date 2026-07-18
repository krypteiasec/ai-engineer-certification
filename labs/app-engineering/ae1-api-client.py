#!/usr/bin/env python3
"""
LAB AE1: Calling model APIs and SDKs. The client wrapper.

An applied AI engineer almost never calls a model with a bare function. You wrap
it in a CLIENT: one object that knows the model, takes a system prompt plus a
list of role/content messages, and hands back a structured response with the
text and the token usage. That shape, a Messages API, is what the Claude SDK and
every OpenAI-compatible endpoint expose, so learning the shape once transfers
everywhere. In this lab you build that client over the shared offline provider
(academy_llm) and prove it returns a real Messages-API-shaped response.

Run: python3 modules/academy-content/labs/app-engineering/ae1-api-client.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


def count_tokens(text):
    # A token here is a whitespace-delimited word. Real SDKs count subword
    # tokens, but a word count is close enough to teach usage accounting.
    return len(text.split())


class Response:
    """Mirrors what a real Messages API returns: a `content` list of typed
    blocks, a `usage` object, and a `stop_reason`. Code downstream reads
    `.text` and `.usage` without caring which provider produced it."""
    def __init__(self, model, text, in_tokens, out_tokens):
        self.model = model
        self.content = [{"type": "text", "text": text}]
        self.usage = {"input_tokens": in_tokens, "output_tokens": out_tokens}
        self.stop_reason = "end_turn"

    @property
    def text(self):
        return "".join(b["text"] for b in self.content if b["type"] == "text")


class Client:
    """A thin client over the model provider (academy_llm is the provider here,
    a real LM Studio or Anthropic endpoint later). ONE method, messages_create,
    shaped like the Claude Messages API: a system prompt plus a list of
    role/content messages in, a structured Response out."""
    def __init__(self, model="academy-mini"):
        self.model = model

    def messages_create(self, system, messages):
        convo = []
        if system:
            convo.append({"role": "system", "content": system})
        convo.extend(messages)
        reply = chat(convo)
        in_text = " ".join(m.get("content", "") for m in convo)
        return Response(self.model, reply, count_tokens(in_text), count_tokens(reply))


# --- Use the client exactly like you would call a real SDK. ------------------
client = Client(model="academy-mini")
resp = client.messages_create(
    system="You are a classifier. You label sentiment.",
    messages=[{"role": "user",
               "content": "Classify the sentiment. Input: I love this product, it works perfectly"}],
)

print("STEP 1: the client returned a structured response, not a bare string")
print("  model        :", resp.model)
print("  content      :", resp.content)
print("  usage        :", resp.usage)
print("  stop_reason  :", resp.stop_reason)
print("  .text        :", repr(resp.text))

# --- Verify the response has the Messages-API shape your app can rely on. ----
shape_ok = (
    isinstance(resp.content, list)
    and resp.content and resp.content[0].get("type") == "text"
    and set(resp.usage.keys()) == {"input_tokens", "output_tokens"}
    and resp.stop_reason == "end_turn"
)
answer_ok = (resp.text == "positive")
usage_ok = (resp.usage["input_tokens"] > 0 and resp.usage["output_tokens"] > 0)

print("")
print("STEP 2: checks")
print("  response has content list + usage + stop_reason :", shape_ok)
print("  the model actually answered correctly           :", answer_ok)
print("  token usage was accounted                        :", usage_ok)

ok = shape_ok and answer_ok and usage_ok
print("")
print("CLIENT WRAPS THE PROVIDER IN MESSAGES-API SHAPE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("One client, one shape, any provider. Next: stream the tokens as they arrive.")
