#!/usr/bin/env python3
"""
LAB AE2: Streaming responses and time-to-first-token.

A model that returns its whole answer at once feels slow, because the user
stares at nothing until the last token lands. Streaming fixes the FELT latency:
the server sends tokens as they are generated (server-sent events in a real API)
and the UI paints them live. Two things matter. The assembled stream must equal
the non-streamed answer exactly, byte for byte, or you have a bug. And
time-to-first-token (TTFT) is the number users feel, not total time. In this lab
you build a streaming simulator over the provider, assemble the tokens back, and
prove the assembled text is identical while measuring a deterministic TTFT.

Run: python3 modules/academy-content/labs/app-engineering/ae2-streaming.py
"""
import sys, os, re, time
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


def stream_tokens(text):
    """Yield the answer in small chunks (delta events). Each chunk keeps its own
    trailing whitespace, so concatenating every chunk reproduces the source text
    exactly. A real SSE stream yields `delta` events the same way."""
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk


def timed_stream(text, first_token_ms=8.0, per_token_ms=3.0):
    """Wrap the token stream with a DETERMINISTIC simulated clock (not wall
    time): the first token costs first_token_ms, each later token costs
    per_token_ms. Yields (chunk, elapsed_ms) so TTFT is measurable offline."""
    clock = 0.0
    first = True
    for chunk in stream_tokens(text):
        clock += first_token_ms if first else per_token_ms
        first = False
        yield chunk, clock


# --- Get one full answer from the provider (the non-streamed baseline). ------
prompt = ("Context: The peregrine falcon is the fastest animal on earth. "
          "The cheetah is the fastest land animal. "
          "Question: What is the fastest animal on earth?")
full = complete(prompt)
print("STEP 1: the full (non-streamed) answer")
print("  ", repr(full))

# --- Stream it token by token and assemble the pieces back. ------------------
assembled = ""
ttft = None
count = 0
for chunk, elapsed in timed_stream(full):
    if ttft is None:
        ttft = elapsed          # first token: this is what the user feels
    assembled += chunk
    count += 1
    time.sleep(0.001)           # a couple of ms, keeps it real but fast

print("")
print("STEP 2: streamed")
print("  tokens streamed      :", count)
print("  time-to-first-token  : %.1f ms (felt latency)" % ttft)
print("  assembled            :", repr(assembled))

# --- An aborted stream is just an early stop: the partial is a real prefix. --
partial = ""
for i, (chunk, _elapsed) in enumerate(timed_stream(full)):
    partial += chunk
    if i == 2:
        break                   # client disconnected / user hit stop
print("")
print("STEP 3: an aborted stream leaves a valid prefix")
print("  partial              :", repr(partial))

identical = (assembled == full)
ttft_beats_total = (ttft < (8.0 + 3.0 * (count - 1)))
prefix_ok = full.startswith(partial)

print("")
print("  assembled equals the full answer :", identical)
print("  TTFT is below total stream time  :", ttft_beats_total)
print("  aborted partial is a real prefix :", prefix_ok)

ok = identical and ttft_beats_total and prefix_ok
print("")
print("STREAMING ASSEMBLED THE SAME TEXT: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Stream for feel, assemble for correctness. Next: structured output you can trust.")
