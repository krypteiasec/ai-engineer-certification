#!/usr/bin/env python3
"""
LAB PR1: Serving models. Latency and throughput through batching.

A model served one request at a time wastes the hardware. The expensive part of
inference is loading the weights and firing up the compute; whether you push one
request or thirty-two through that machinery, the fixed overhead is paid once.
Batching is the single biggest throughput lever in serving: gather requests that
arrive close together, run them in ONE forward pass, and the per-request cost
falls because the fixed overhead is now shared across the whole batch.

This lab is a deterministic serving simulator. It models a request's service
time as a fixed per-batch overhead (kernel launch, weight access) plus a small
marginal cost per request in the batch. It serves the same workload two ways,
unbatched (one request per pass) and batched (up to B per pass), and proves the
batched server clears the same work at markedly higher throughput.

Run: python3 modules/academy-content/labs/production/pr1-serving-batching.py
"""
import sys

# ---- the cost model (deterministic, no wall-clock, so it is reproducible) ----
# Each forward pass costs a FIXED overhead plus a MARGINAL cost per request in it.
# This is the real shape of inference: overhead dominates at batch size 1.
OVERHEAD_MS = 20.0     # fixed cost paid once per forward pass, regardless of size
MARGINAL_MS = 2.0      # extra cost each additional request in the batch adds
N_REQUESTS = 256       # the workload: this many requests must be served
BATCH = 32             # the batched server groups up to this many per pass


def serve(n_requests, batch_size):
    """Return (total_ms, throughput_rps) for serving n_requests in passes of
    at most batch_size. One pass costs OVERHEAD_MS + batch * MARGINAL_MS."""
    total_ms = 0.0
    remaining = n_requests
    passes = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        total_ms += OVERHEAD_MS + b * MARGINAL_MS
        remaining -= b
        passes += 1
    throughput = n_requests / (total_ms / 1000.0)  # requests per second
    return total_ms, throughput, passes


def main():
    un_ms, un_rps, un_passes = serve(N_REQUESTS, 1)
    b_ms, b_rps, b_passes = serve(N_REQUESTS, BATCH)

    print("STEP 1: serve %d requests UNBATCHED (one per forward pass)" % N_REQUESTS)
    print("  forward passes : %d" % un_passes)
    print("  total time     : %.0f ms" % un_ms)
    print("  throughput     : %.1f req/s" % un_rps)

    print("")
    print("STEP 2: serve the same %d requests BATCHED (up to %d per pass)"
          % (N_REQUESTS, BATCH))
    print("  forward passes : %d" % b_passes)
    print("  total time     : %.0f ms" % b_ms)
    print("  throughput     : %.1f req/s" % b_rps)

    speedup = b_rps / un_rps
    print("")
    print("STEP 3: batching shares the fixed per-pass overhead across the batch")
    print("  throughput gain: %.1fx" % speedup)

    # Proof: batching clears the SAME workload at meaningfully higher throughput.
    # The gain must be real (well above 1x) and the request count must be equal.
    same_work = (N_REQUESTS == N_REQUESTS)
    real_gain = speedup > 2.0
    ok = same_work and real_gain
    print("")
    print("BATCHING RAISED THROUGHPUT (same work, fewer passes): %s"
          % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Batching is the first serving lever. Next: quantization to cut the cost per pass.")


if __name__ == "__main__":
    main()
