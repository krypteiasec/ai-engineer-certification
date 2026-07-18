#!/usr/bin/env python3
"""
LAB PR3: The KV cache and the memory cost of serving.

Generation is autoregressive: the model produces one token at a time, and every
new token attends back over every previous token. Recomputing the keys and values
for the whole history on each step would be quadratic and hopeless. The KV cache
fixes the speed by storing each token's key and value once and reusing them, which
is why serving is fast. But that cache is not free: it lives in memory, and it
grows LINEARLY with the sequence length, per layer, per attention head. On a busy
server the KV cache, not the weights, is often what runs you out of memory and caps
how many concurrent requests (and how long a context) you can serve.

This lab builds the KV-cache memory model from first principles. The bytes are:
    2 (K and V) * layers * heads * head_dim * seq_len * dtype_bytes
It computes the cache size across growing context lengths, proves the growth is
exactly linear in context length, and shows the concurrency limit a memory budget
implies. This is the arithmetic behind "why can't I serve a 128k context to 100
users at once."

Run: python3 modules/academy-content/labs/production/pr3-kv-cache-cost.py
"""
import sys

# A model config, roughly a small-to-mid transformer.
LAYERS = 32
HEADS = 32
HEAD_DIM = 128
DTYPE_BYTES = 2          # fp16/bf16 KV cache, the common serving choice
GB = 1024 ** 3


def kv_bytes(seq_len):
    """KV-cache bytes for ONE request at a given context length."""
    return 2 * LAYERS * HEADS * HEAD_DIM * seq_len * DTYPE_BYTES


def main():
    lengths = [1024, 2048, 4096, 8192, 16384]
    print("STEP 1: KV-cache memory per request, by context length")
    print("  (2 * layers %d * heads %d * head_dim %d * seq_len * %d bytes)"
          % (LAYERS, HEADS, HEAD_DIM, DTYPE_BYTES))
    sizes = []
    for L in lengths:
        b = kv_bytes(L)
        sizes.append(b)
        print("  seq_len %6d : %.3f GB per request" % (L, b / GB))

    # ---- prove linearity: doubling context doubles the cache, exactly ----
    print("")
    print("STEP 2: doubling the context doubles the cache (linear growth)")
    per_token = kv_bytes(1)
    linear = True
    for L in lengths:
        expected = per_token * L            # linear model
        actual = kv_bytes(L)
        if expected != actual:
            linear = False
        print("  seq_len %6d : %d bytes, per-token * len = %d, match %s"
              % (L, actual, expected, actual == expected))
    ratio = kv_bytes(2048) / kv_bytes(1024)
    print("  cache(2048) / cache(1024) = %.2f (expect exactly 2.0)" % ratio)

    # ---- the concurrency ceiling a memory budget implies ----
    print("")
    print("STEP 3: how many concurrent 8k-context requests fit in a budget")
    budget_gb = 40.0                        # e.g. leftover VRAM after weights
    per_req_gb = kv_bytes(8192) / GB
    max_concurrent = int(budget_gb / per_req_gb)
    print("  budget %.0f GB, per-request (8k ctx) %.3f GB -> %d concurrent requests"
          % (budget_gb, per_req_gb, max_concurrent))

    # ---- proofs ----
    linear_ok = linear and abs(ratio - 2.0) < 1e-9
    budget_ok = max_concurrent >= 1 and per_req_gb * (max_concurrent + 1) > budget_gb
    ok = linear_ok and budget_ok
    print("")
    print("  growth is linear in context: %s   budget math holds: %s"
          % ("YES" if linear_ok else "NO", "YES" if budget_ok else "NO"))
    print("")
    print("KV-CACHE MEMORY GROWS LINEARLY WITH CONTEXT: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("The cache, not the weights, caps concurrency. Next: gate deploys on evals.")


if __name__ == "__main__":
    main()
