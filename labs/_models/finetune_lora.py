#!/usr/bin/env python3
"""
Real LoRA fine-tune ON the trained tinyGPT (Course 7).

Loads the frozen base checkpoint, attaches low-rank adapters (A/B) to the query
and value projections of every attention block, and trains ONLY those adapters.
The downstream task is the classic LoRA use case: adapt a general model to one
narrow subdomain. Here the subdomain is SECURITY / VERIFICATION prose.

The effect is visible and assertable on a fixed prompt:
  BASE  ("a neural network is ...") keeps talking about layers and tools.
  LoRA  ("a neural network is ...") steers the SAME frozen weights toward
        security and verification: check inputs, trust nothing, verify by running.

Base weights are frozen; only the adapters (a few thousand numbers) train.

Output (committed artifact):
  lora_adapter.pt  -> just the adapter tensors (tiny)

Run: python3 finetune_lora.py
"""
import os
import json
import time
import torch
import torch.nn.functional as F

from model import (TinyGPT, CharTokenizer, generate,
                   inject_lora, lora_state_dict)

HERE = os.path.dirname(os.path.abspath(__file__))
FINETUNE = os.path.join(HERE, "corpus_finetune.txt")
WEIGHTS = os.path.join(HERE, "tinygpt.pt")
META = os.path.join(HERE, "tinygpt_meta.json")
ADAPTER = os.path.join(HERE, "lora_adapter.pt")

BATCH = 48
STEPS = 2000
LR = 3e-3
LORA_R = 16
LORA_ALPHA = 32
SEED = 1234
PROMPT = "a neural network is"

# words that mark the security/verification subdomain the LoRA is taught.
THEME_WORDS = ["verify", "trust", "input", "secur", "check", "test", "attacker", "proof", "run"]


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_batch(data, n_block, device):
    ix = torch.randint(0, len(data) - n_block - 1, (BATCH,))
    x = torch.stack([data[i:i + n_block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + n_block] for i in ix])
    return x.to(device), y.to(device)


def theme_hits(s):
    low = s.lower()
    return sum(low.count(w) for w in THEME_WORDS)


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    meta = json.load(open(META))
    tok = CharTokenizer(list(meta["vocab"]))
    V = meta["vocab_size"]

    # base output BEFORE adapting, for the before/after comparison.
    base = TinyGPT(V, meta["c"], meta["n_layer"], meta["n_block"])
    base.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    base_out = generate(base, tok, PROMPT, n=200, temperature=0.0, device="cpu")

    # fresh model, load base weights, inject LoRA, freeze base, train adapters.
    model = TinyGPT(V, meta["c"], meta["n_layer"], meta["n_block"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    lora_params = inject_lora(model, r=LORA_R, alpha=LORA_ALPHA)
    model.to(device)

    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print("device: %s | trainable LoRA params: %d of %d (%.2f%%)"
          % (device, n_lora, n_total, 100.0 * n_lora / n_total))

    fine = open(FINETUNE).read()
    data = torch.tensor(tok.encode(fine), dtype=torch.long)

    opt = torch.optim.AdamW(lora_params, lr=LR)
    losses = []
    t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(data, meta["n_block"], device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == STEPS - 1:
            print("  step %4d  loss %.4f  (%.1fs)" % (step, loss.item(), time.time() - t0))

    print("")
    print("LoRA loss fell from %.4f to %.4f" % (losses[0], losses[-1]))

    torch.save(lora_state_dict(model), ADAPTER)
    print("saved %s (%d bytes)" % (ADAPTER, os.path.getsize(ADAPTER)))

    # after: adapted model on the SAME prompt.
    model_cpu = model.to("cpu")
    lora_out = generate(model_cpu, tok, PROMPT, n=200, temperature=0.0, device="cpu")

    print("")
    print("PROMPT: '%s'" % PROMPT)
    print("BASE  (temp 0): " + base_out.replace("\n", " ")[:170])
    print("LoRA  (temp 0): " + lora_out.replace("\n", " ")[:170])
    print("")
    b_hits, l_hits = theme_hits(base_out), theme_hits(lora_out)
    print("security/verification theme hits -> base: %d | LoRA: %d" % (b_hits, l_hits))
    differs = base_out != lora_out
    shifted = l_hits >= 3 and l_hits > b_hits
    print("outputs differ: %s | subdomain shift learned: %s"
          % ("YES" if differs else "NO", "YES" if shifted else "NO"))
    if not (differs and shifted):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
