#!/usr/bin/env python3
"""
LAB TR5: The real thing. A pretrained model, base vs LoRA, same frozen weights.

TR4 built a LoRA layer from scratch on a toy. This lab uses the Academy's REAL
trained artifacts: a from-scratch TinyGPT (about 848K parameters) trained on a
broad AI/programming corpus, plus a real LoRA adapter (about 134 KB, roughly 4%
of the parameters) trained on a narrow SECURITY / VERIFICATION subdomain with the
base frozen. Nothing trains here: you LOAD both and watch what LoRA does.

The point of LoRA is that one frozen base plus a tiny swappable adapter gives you
a specialized model without a second copy of the weights. You will see it: on the
exact same prompt, the base keeps talking about layers and tokens, while the
LoRA-adapted model steers the SAME frozen weights toward security and
verification prose (check inputs, trust nothing, verify by running).

It PROVES the adaptation two ways: the outputs differ, and the LoRA output hits
the security/verification vocabulary clearly more than the base does.

Fast: it loads checkpoints and generates deterministically, about a second.

Run: python3 modules/academy-content/labs/training/tr5-lora-base-vs-adapter.py
"""
import sys, os
_labs = None
for _c in [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
           os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]:
    if _c and os.path.isdir(os.path.join(_c, "_models")): _labs = os.path.abspath(_c); break
if _labs: sys.path.insert(0, os.path.join(_labs, "_models"))

from load import load_tinygpt, load_tinygpt_with_lora, generate

PROMPT = "a neural network is"
# words that mark the security/verification subdomain the adapter was trained on.
THEME = ["verify", "trust", "input", "secur", "check", "test", "attacker", "proof", "run"]


def theme_hits(s):
    low = s.lower()
    return sum(low.count(w) for w in THEME)


def main():
    print("STEP 1: load the pretrained base TinyGPT (frozen weights on disk)")
    base = load_tinygpt()
    base_out = generate(base, PROMPT, n=180).replace("\n", " ")
    print("  BASE: " + base_out[:150])

    print("")
    print("STEP 2: load the SAME base with the trained LoRA adapter attached")
    lora = load_tinygpt_with_lora()
    lora_out = generate(lora, PROMPT, n=180).replace("\n", " ")
    print("  LoRA: " + lora_out[:150])

    b_hits = theme_hits(base_out)
    l_hits = theme_hits(lora_out)
    print("")
    print("STEP 3: measure the domain shift on a fixed prompt")
    print("  security/verification theme hits -> base: %d | LoRA: %d" % (b_hits, l_hits))

    differ = base_out != lora_out
    shifted = l_hits >= 3 and l_hits > b_hits
    ok = differ and shifted
    print("  outputs differ: %s | LoRA hits the new domain more: %s"
          % ("YES" if differ else "NO", "YES" if shifted else "NO"))
    print("")
    print("LORA ADAPTED THE FROZEN BASE (output shifted to the new domain): %s"
          % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
