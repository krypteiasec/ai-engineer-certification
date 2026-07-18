#!/usr/bin/env python3
"""
LAB SDK3: Streaming responses.

A model that returns its whole answer at once feels slow, because the user stares
at a blank screen until the last token lands. Streaming fixes the FELT latency:
the server sends tokens as they are generated and the UI paints them live. In the
real SDK you switch from create to stream:

    with client.messages.stream(
        model="claude-opus-4-8", max_tokens=1024,
        messages=[{"role": "user", "content": "Write a haiku about the sea."}],
    ) as stream:
        for text in stream.text_stream:      # incremental deltas as they arrive
            print(text, end="", flush=True)
        final = stream.get_final_message()   # the complete, assembled Message

Under the hood the wire format is server-sent events: message_start, then
content_block_delta events each carrying a text_delta, then message_stop. Two
truths. The assembled stream must equal the non-streamed answer exactly, or you
have a bug. And time-to-first-token (TTFT), not total time, is the number a user
actually feels. Streaming is also REQUIRED for large max_tokens (above ~16K),
because a single blocking request would hit an HTTP timeout. In this lab you
build a delta stream over the provider, assemble it back, and prove it is byte
identical while measuring a deterministic TTFT.

Run: python3 modules/academy-content/labs/claude-code-sdk/sdk3-streaming.py
"""
import sys, os, re, time
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete


def text_deltas(text):
    """Yield the answer as content_block_delta events. Each delta keeps its own
    trailing whitespace so concatenating every delta reproduces the source text
    exactly, the way a real SSE text_stream does."""
    for chunk in re.findall(r"\S+\s*", text):
        yield chunk


def timed_stream(text, first_token_ms=8.0, per_token_ms=3.0):
    """Wrap the delta stream with a DETERMINISTIC simulated clock (not wall time):
    the first token costs first_token_ms, each later token costs per_token_ms.
    Yields (delta, elapsed_ms) so TTFT is measurable offline and reproducibly."""
    clock, first = 0.0, True
    for chunk in text_deltas(text):
        clock += first_token_ms if first else per_token_ms
        first = False
        yield chunk, clock


# --- The non-streamed baseline (what get_final_message would return). --------
prompt = ("Context: The peregrine falcon is the fastest animal on earth. "
          "The cheetah is the fastest land animal. "
          "Question: What is the fastest animal on earth?")
full = complete(prompt)
print("STEP 1: the full (non-streamed) message")
print("  ", repr(full))

# --- Stream the deltas and assemble them back. -------------------------------
assembled, ttft, count = "", None, 0
for delta, elapsed in timed_stream(full):
    if ttft is None:
        ttft = elapsed               # first token: this is what the user feels
    assembled += delta
    count += 1
    time.sleep(0.001)               # a hair of real time, keeps it honest and fast

print("")
print("STEP 2: streamed")
print("  deltas streamed      :", count)
print("  time-to-first-token  : %.1f ms (felt latency)" % ttft)
print("  assembled            :", repr(assembled))

# --- An aborted stream is just an early stop: the partial is a real prefix. ---
partial = ""
for i, (delta, _e) in enumerate(timed_stream(full)):
    partial += delta
    if i == 2:
        break                        # client disconnected or user hit stop
print("")
print("STEP 3: an aborted stream leaves a valid prefix")
print("  partial              :", repr(partial))

identical = (assembled == full)
ttft_beats_total = (ttft < (8.0 + 3.0 * (count - 1)))
prefix_ok = full.startswith(partial)

print("")
print("  assembled equals the full message :", identical)
print("  TTFT is below total stream time   :", ttft_beats_total)
print("  aborted partial is a real prefix  :", prefix_ok)

ok = identical and ttft_beats_total and prefix_ok
print("")
print("STREAMED DELTAS ASSEMBLE TO THE FULL MESSAGE: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Stream for feel, assemble for correctness. Next: give the model tools.")
