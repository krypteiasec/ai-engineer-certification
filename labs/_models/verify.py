#!/usr/bin/env python3
"""
Self-test for the trained Academy model artifacts.

Checks, in order:
  1. base checkpoint loads, generates non-empty text, and is DETERMINISTIC at
     temperature 0 (same prompt -> identical text on two independent runs);
  2. LoRA checkpoint loads and, on the fine-tune prompt, generates text that
     DIFFERS from the base and carries the security/verification subdomain it
     was trained on (theme words the base output does not use).

Prints MODELS OK and exits 0 on success; raises and exits non-zero otherwise.

Run: python3 verify.py
"""
import sys

from load import load_tinygpt, load_tinygpt_with_lora, generate

PROMPT = "a neural network is"
THEME_WORDS = ["verify", "trust", "input", "secur", "check", "test", "attacker", "proof", "run"]


def theme_hits(s):
    low = s.lower()
    return sum(low.count(w) for w in THEME_WORDS)


def main():
    # 1. base: loads, non-empty, deterministic at temp 0.
    base = load_tinygpt()
    out_a = generate(base, PROMPT, n=180, temperature=0.0)
    out_b = generate(base, PROMPT, n=180, temperature=0.0)
    assert out_a, "base generated empty output"
    assert len(out_a) > len(PROMPT), "base produced no continuation"
    assert out_a == out_b, "base is not deterministic at temperature 0"
    print("[1] base loads, generates, deterministic at temp 0  OK")
    print("    base: " + out_a.replace("\n", " ")[:150])

    # 2. LoRA: loads, differs from base, carries the trained subdomain.
    lora = load_tinygpt_with_lora()
    lora_out = generate(lora, PROMPT, n=180, temperature=0.0)
    assert lora_out, "lora generated empty output"
    assert lora_out != out_a, "lora output does not differ from base"
    b_hits, l_hits = theme_hits(out_a), theme_hits(lora_out)
    assert l_hits >= 3, "lora output missing the trained subdomain (hits=%d)" % l_hits
    assert l_hits > b_hits, "lora did not shift toward the subdomain vs base (%d vs %d)" % (l_hits, b_hits)
    print("[2] lora loads, differs from base, subdomain present  OK")
    print("    lora: " + lora_out.replace("\n", " ")[:150])
    print("    theme hits  base: %d  lora: %d" % (b_hits, l_hits))

    print("")
    print("MODELS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
