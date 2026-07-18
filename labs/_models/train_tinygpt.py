#!/usr/bin/env python3
"""
Train the Course-1 tinyGPT from scratch on the local corpus.

This is the lab-08 training loop (forward -> loss -> backward -> step -> repeat),
scaled from the toy bigram up to the full lab-07 transformer, run for real over
a few hundred KB of text. It uses MPS if available, else CPU, and finishes in a
couple of minutes.

Outputs (committed artifacts):
  tinygpt.pt         -> trained weights (state_dict)
  tinygpt_meta.json  -> vocab string + config so the loader can rebuild the model

Run: python3 train_tinygpt.py
"""
import os
import json
import time
import torch
import torch.nn.functional as F

from model import TinyGPT, CharTokenizer, generate

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")
WEIGHTS = os.path.join(HERE, "tinygpt.pt")
META = os.path.join(HERE, "tinygpt_meta.json")

# Config: same architecture as lab-07, scaled up enough to actually learn.
C = 128        # embedding width
N_LAYER = 4    # transformer blocks
N_BLOCK = 128  # context window
BATCH = 48
STEPS = 1500
LR = 3e-3
SEED = 1234


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, device):
    ix = torch.randint(0, len(data) - N_BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + N_BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + N_BLOCK] for i in ix])
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(SEED)
    device = pick_device()

    text = open(CORPUS).read()
    # vocab is the character set of the base corpus. The LoRA fine-tune corpus is
    # built to use only these characters, so the same tokenizer serves both with
    # no remapping and no embedding-table resize on the frozen base.
    vocab_chars = sorted(set(text))
    tok = CharTokenizer(vocab_chars)
    V = tok.vocab_size

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print("device: %s | corpus: %d chars | vocab: %d | params model C=%d L=%d ctx=%d"
          % (device, len(data), V, C, N_LAYER, N_BLOCK))

    model = TinyGPT(V, C, N_LAYER, N_BLOCK).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("trainable parameters: %d (%.2fK)" % (n_params, n_params / 1e3))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(data, device)
        logits = model(x)                       # (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == STEPS - 1:
            print("  step %4d  loss %.4f  (%.1fs)" % (step, loss.item(), time.time() - t0))

    final = sum(losses[-20:]) / 20
    print("")
    print("loss fell from %.4f to %.4f (20-step avg %.4f)" % (losses[0], losses[-1], final))
    learned = final < losses[0] * 0.6
    print("model LEARNED (final avg < 60%% of initial): %s" % ("YES" if learned else "NO"))

    # save weights + meta
    torch.save(model.state_dict(), WEIGHTS)
    meta = {
        "vocab": "".join(vocab_chars),
        "c": C,
        "n_layer": N_LAYER,
        "n_block": N_BLOCK,
        "vocab_size": V,
        "steps": STEPS,
        "final_loss": final,
    }
    with open(META, "w") as f:
        json.dump(meta, f, indent=2)
    print("saved %s (%d bytes) and %s" % (WEIGHTS, os.path.getsize(WEIGHTS), META))

    # sample generation at temperature 0 (deterministic)
    model_cpu = model.to("cpu")
    sample = generate(model_cpu, tok, "a neural network is", n=220, temperature=0.0, device="cpu")
    print("")
    print("SAMPLE (temp 0, prompt 'a neural network is'):")
    print("  " + sample.replace("\n", " "))

    if not learned:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
