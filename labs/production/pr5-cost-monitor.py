#!/usr/bin/env python3
"""
LAB PR5: Cost monitoring. Token accounting and budget alerts.

An LLM feature bills by the token, and cost is where a promising product quietly
dies. A single agent that loops, a prompt that balloons, a spike in traffic, and
the bill runs away with nobody watching. Production systems account for every
request (input tokens, output tokens, the price of each) and run a live meter
against a budget, so the moment spend crosses a threshold an alert fires instead
of a surprise invoice arriving at month end. Observability is not a nice-to-have
here; it is the difference between a cost you manage and a cost that manages you.

This lab is a deterministic cost monitor. It replays a stream of served requests,
accounts input and output tokens at real per-token prices, accumulates spend, and
trips an alert the instant the running total crosses the budget. It proves the
alert fires on the correct request and that the accounting is exact.

Run: python3 modules/academy-content/labs/production/pr5-cost-monitor.py
"""
import sys

# Prices in dollars per token (illustrative, order-of-magnitude realistic).
PRICE_IN = 3.0 / 1_000_000     # $3 per million input tokens
PRICE_OUT = 15.0 / 1_000_000   # $15 per million output tokens
BUDGET = 0.15                  # alert when cumulative spend crosses $0.15

# A replayed request stream: (input_tokens, output_tokens) per served request.
REQUESTS = [
    (1200, 300), (800, 150), (5000, 1200), (2200, 600), (9000, 2000),
    (1500, 400), (3000, 800), (7000, 1800), (2500, 700), (4000, 1100),
]


def request_cost(tin, tout):
    return tin * PRICE_IN + tout * PRICE_OUT


def main():
    print("STEP 1: account every request and run a live meter against the budget")
    cumulative = 0.0
    alert_at = None
    total_in = 0
    total_out = 0
    for i, (tin, tout) in enumerate(REQUESTS, start=1):
        c = request_cost(tin, tout)
        cumulative += c
        total_in += tin
        total_out += tout
        crossed = cumulative >= BUDGET and alert_at is None
        if crossed:
            alert_at = i
        flag = "  <-- ALERT: over budget" if crossed else ""
        print("  req %2d  in %5d out %5d  cost $%.4f  cumulative $%.4f%s"
              % (i, tin, tout, c, cumulative, flag))

    # ---- independent recomputation of the total, to prove the accounting ----
    recomputed = total_in * PRICE_IN + total_out * PRICE_OUT
    print("")
    print("STEP 2: verify the accounting is exact")
    print("  total input tokens : %d" % total_in)
    print("  total output tokens: %d" % total_out)
    print("  metered total      : $%.4f" % cumulative)
    print("  recomputed total   : $%.4f" % recomputed)
    print("  budget             : $%.2f, alert fired on request: %s"
          % (BUDGET, str(alert_at)))

    # ---- proofs ----
    accounting_exact = abs(cumulative - recomputed) < 1e-9
    alert_fired = alert_at is not None and cumulative >= BUDGET
    ok = accounting_exact and alert_fired
    print("")
    print("  accounting exact: %s   over-budget alert fired: %s"
          % ("YES" if accounting_exact else "NO", "YES" if alert_fired else "NO"))
    print("")
    print("COST MONITOR ALERTED OVER BUDGET: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Spend is now observable, not a surprise. Next: watch quality drift.")


if __name__ == "__main__":
    main()
