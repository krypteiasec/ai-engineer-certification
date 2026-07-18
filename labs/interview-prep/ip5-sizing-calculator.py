#!/usr/bin/env python3
"""
LAB IP5: An LLM system-design sizing calculator.

The system-design round is where offers are won and lost, and the thing that
separates a strong answer from hand-waving is NUMBERS. When you say "we will
batch on one A100" or "this costs about X per thousand requests", the follow-up
is always "show me the math." This lab is that math, as three functions every
LLM system designer should be able to derive on the spot:

  1. KV-cache memory: 2 * layers * heads * head_dim * seq_len * batch * bytes.
     The 2 is for K and V. This is the memory that limits how many concurrent
     requests fit on a GPU, so it drives your batching design.
  2. Generation latency: output_tokens / tokens_per_second. Decode is
     sequential, so output length, not input length, dominates wall-clock time.
  3. Cost per 1000 requests: priced separately for input and output tokens,
     because output tokens usually cost several times more than input tokens.

You implement all three and PROVE them against worked numbers computed by hand.

Run: python3 modules/academy-content/labs/interview-prep/ip5-sizing-calculator.py
"""
import sys

GIB = 1024 ** 3


def kv_cache_bytes(layers, heads, head_dim, seq_len, batch, dtype_bytes):
    """Bytes of KV cache. Factor of 2 because we store BOTH keys and values."""
    return 2 * layers * heads * head_dim * seq_len * batch * dtype_bytes


def gen_latency_seconds(output_tokens, tokens_per_second):
    """Decode is one token at a time, so time is output length over throughput.
    Input tokens are processed in one parallel prefill and are not the
    bottleneck for latency here."""
    return output_tokens / tokens_per_second


def cost_per_1k_requests(avg_input_tokens, avg_output_tokens,
                         price_in_per_mtok, price_out_per_mtok):
    """Cost of 1000 requests, pricing input and output tokens separately (per
    million tokens, the standard billing unit)."""
    per_request = (avg_input_tokens / 1_000_000 * price_in_per_mtok
                   + avg_output_tokens / 1_000_000 * price_out_per_mtok)
    return per_request * 1000


# ---- Worked example 1: KV cache for a 7B-ish config -----------------------
# 32 layers, 32 heads, head_dim 128, seq 2048, batch 1, fp16 (2 bytes each).
# 2 * 32 * 32 * 128 * 2048 * 1 * 2 = 1,073,741,824 bytes = exactly 1.0 GiB.
kv = kv_cache_bytes(layers=32, heads=32, head_dim=128, seq_len=2048,
                    batch=1, dtype_bytes=2)
kv_gib = kv / GIB
print("STEP 1: KV-cache memory")
print(f"  bytes = {kv:,}")
print(f"  = {kv_gib:.4f} GiB  (worked answer: 1.0 GiB)")
kv_ok = (kv == 1_073_741_824) and abs(kv_gib - 1.0) < 1e-9

# ---- Worked example 2: generation latency ---------------------------------
# 500 output tokens at 2000 tokens/sec = 0.25 seconds.
lat = gen_latency_seconds(output_tokens=500, tokens_per_second=2000)
print("")
print("STEP 2: generation latency")
print(f"  500 tokens / 2000 tok per sec = {lat:.4f} s  (worked answer: 0.25 s)")
lat_ok = abs(lat - 0.25) < 1e-12

# ---- Worked example 3: cost per 1000 requests -----------------------------
# 1000 input tokens at $3/Mtok  = $0.003 per request.
# 500 output tokens at $15/Mtok = $0.0075 per request.
# per request = $0.0105  ->  per 1000 requests = $10.50.
cost = cost_per_1k_requests(avg_input_tokens=1000, avg_output_tokens=500,
                            price_in_per_mtok=3.0, price_out_per_mtok=15.0)
print("")
print("STEP 3: cost per 1000 requests")
print(f"  = ${cost:.2f}  (worked answer: $10.50)")
cost_ok = abs(cost - 10.50) < 1e-9

print("")
print(f"        KV-cache math correct  : {kv_ok}")
print(f"        latency math correct   : {lat_ok}")
print(f"        cost math correct      : {cost_ok}")

ok = kv_ok and lat_ok and cost_ok
print("")
print(f"SYSTEM-DESIGN SIZING MATCHES WORKED NUMBERS: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Numbers beat hand-waving in the design round. Next round: the take-home.")
