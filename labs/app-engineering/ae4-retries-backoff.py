#!/usr/bin/env python3
"""
LAB AE4: Resilience. Retries with exponential backoff and jitter.

Production APIs fail transiently: a 429 rate limit, a 503, a dropped connection.
None of these mean your request is wrong, they mean try again in a moment. The
correct pattern is exponential backoff with jitter: wait a little, then double
the wait each attempt, and add a random slice so a fleet of clients does not
retry in lockstep and hammer the server (the thundering herd). In this lab a
provider fails transiently a fixed number of times, then succeeds, and a
retry_with_backoff wrapper rides through the failures and returns the answer.

Run: python3 modules/academy-content/labs/app-engineering/ae4-retries-backoff.py
"""
import sys, os, time, random
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import complete, chat, embed, cosine


class TransientError(Exception):
    """A retryable failure, like a 429 or a 503. Non-transient errors (a bad
    request) should NOT be retried, which is why we catch this type only."""


class FlakyProvider:
    """Wraps the real provider but injects a fixed number of transient failures
    first, then succeeds. Deterministic on purpose: the failures are scripted,
    not random, so the lab proves the recovery every single run."""
    def __init__(self, fail_times):
        self.remaining_failures = fail_times
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise TransientError("429 rate limited (call %d)" % self.calls)
        return complete(prompt)


def retry_with_backoff(fn, max_attempts=5, base_ms=10.0, seed=0):
    """Call fn(); on a TransientError wait base * 2**attempt plus full jitter,
    then try again, up to max_attempts. Returns (result, delays_ms). Real sleeps
    are capped to a couple of ms here so the lab is fast; the SCHEDULE printed is
    the real one you would use in production."""
    rng = random.Random(seed)
    delays = []
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn(), delays
        except TransientError as e:
            last_err = e
            backoff = base_ms * (2 ** attempt)     # exponential
            jitter = rng.uniform(0.0, base_ms)     # full jitter, spreads the herd
            wait = round(backoff + jitter, 2)
            delays.append(wait)
            print("  attempt %d failed (%s) -> backoff %.2f ms" % (attempt + 1, e, wait))
            time.sleep(min(wait, 2.0) / 1000.0)    # capped real sleep (<=2ms)
    raise last_err


provider = FlakyProvider(fail_times=2)

print("STEP 1: call through the retry wrapper (provider will fail twice first)")
answer, delays = retry_with_backoff(
    lambda: provider.complete("Classify the sentiment. Input: this release is fantastic"),
    max_attempts=5, base_ms=10.0, seed=0,
)

print("")
print("STEP 2: result")
print("  provider calls made :", provider.calls)
print("  transient failures  :", len(delays))
print("  backoff schedule ms :", delays)
print("  final answer        :", repr(answer))

recovered = (answer == "positive")
two_failures = (len(delays) == 2 and provider.calls == 3)
grows = (len(delays) == 2 and delays[1] > delays[0])   # exponential => second wait bigger

print("")
print("  call eventually succeeded    :", recovered)
print("  survived exactly 2 failures  :", two_failures)
print("  backoff grew each attempt    :", grows)

ok = recovered and two_failures and grows
print("")
print("BACKOFF RECOVERED AFTER 2 TRANSIENT FAILURES: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Retry transient errors, back off exponentially, add jitter. Next: control the cost.")
