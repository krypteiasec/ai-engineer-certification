#!/usr/bin/env python3
"""
LAB SDK7: Prompt caching and token counting.

You pay per token, and most apps quietly overpay. They resend the same long
system prompt on every request and pay full price for it every time. PROMPT
CACHING stores a stable prefix after the first call, and later calls that reuse
that prefix pay a fraction for it. You mark the cache point with cache_control:

    resp = client.messages.create(
        model="claude-opus-4-8", max_tokens=1024,
        system=[{"type": "text", "text": BIG_SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": question}],
    )
    resp.usage.cache_creation_input_tokens   # written to cache (first call, ~1.25x)
    resp.usage.cache_read_input_tokens       # served from cache (later calls, ~0.1x)
    resp.usage.input_tokens                  # uncached remainder (full price)

Caching is a PREFIX match: render order is tools, then system, then messages, so
put stable content first (frozen system prompt) and volatile content last (the
changing question). Any byte change in the prefix invalidates the cache. To size
a prompt before you send it, use the token counting endpoint, never a third-party
tokenizer:

    n = client.messages.count_tokens(model="claude-opus-4-8", messages=[...]).input_tokens

In this lab you run one stable system prompt across several questions, account
tokens with cache-read versus cache-creation, and prove the repeated cost falls
while the answers stay identical, plus that count_tokens agrees with what you sent.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk7-caching-tokens.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


def count_tokens(text):
    """The token counting endpoint, in miniature. Real Claude counts subword
    tokens; a word count teaches the accounting. NEVER use a non-Claude
    tokenizer to size a Claude prompt: it will be wrong."""
    return len(text.split())


BIG_SYSTEM = ("You are a sentiment classifier. Read each review and label it "
              "positive, negative, or neutral. Return only the label. " * 6)   # a big stable prefix
QUESTIONS = [
    "sentiment: I love this product",
    "sentiment: this update is terrible",
    "sentiment: it is fine, nothing special",
    "sentiment: absolutely fantastic experience",
]


class Usage:
    def __init__(self, input_tokens, cache_creation, cache_read):
        self.input_tokens = input_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read

    @property
    def billed(self):
        # Cache writes cost ~1.25x, reads ~0.1x, fresh input 1x. We use those
        # multipliers so the SAVINGS are visible in one comparable number.
        return self.input_tokens + 1.25 * self.cache_creation_input_tokens \
            + 0.1 * self.cache_read_input_tokens


def call(system, question, cache, cached_prefix):
    """Run one call. If caching is on and the system prefix was already written,
    those tokens are billed as a cheap cache_read; otherwise as cache_creation."""
    sys_tokens = count_tokens(system)
    q_tokens = count_tokens(question)
    answer = complete(system + "\n" + question)
    if not cache:
        return answer, Usage(sys_tokens + q_tokens, 0, 0), cached_prefix
    if cached_prefix is None:
        return answer, Usage(q_tokens, sys_tokens, 0), system     # first call writes cache
    return answer, Usage(q_tokens, 0, sys_tokens), cached_prefix   # later calls read cache


# --- Run WITHOUT caching. ----------------------------------------------------
no_cache_billed, no_cache_answers = 0.0, []
for q in QUESTIONS:
    ans, usage, _ = call(BIG_SYSTEM, q, cache=False, cached_prefix=None)
    no_cache_billed += usage.billed
    no_cache_answers.append(ans)

# --- Run WITH caching (prefix written once, read after). ---------------------
cache_billed, cache_answers, prefix = 0.0, [], None
for q in QUESTIONS:
    ans, usage, prefix = call(BIG_SYSTEM, q, cache=True, cached_prefix=prefix)
    cache_billed += usage.billed
    cache_answers.append(ans)

print("STEP 1: same system prompt, four questions")
print("  answers (no cache) :", no_cache_answers)
print("  answers (cache)    :", cache_answers)
print("")
print("STEP 2: billed tokens")
print("  without caching    : %.1f" % no_cache_billed)
print("  with caching       : %.1f" % cache_billed)

# --- Token counting agrees with what we sent. --------------------------------
probe = BIG_SYSTEM + "\n" + QUESTIONS[0]
counted = count_tokens(probe)
print("")
print("STEP 3: count_tokens before sending")
print("  count_tokens(probe):", counted, "tokens")

answers_identical = (no_cache_answers == cache_answers)
cache_is_cheaper = (cache_billed < no_cache_billed)
count_matches = (counted == len(probe.split()) and counted > 0)

print("")
print("  caching left the answers unchanged   :", answers_identical)
print("  caching lowered the repeated cost     :", cache_is_cheaper)
print("  count_tokens matched the sent prompt  :", count_matches)

ok = answers_identical and cache_is_cheaper and count_matches
print("")
print("PROMPT CACHING CUTS REPEATED TOKENS, COUNT MATCHES: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Stable prefix first, measure with count_tokens. Next: build the whole app.")
