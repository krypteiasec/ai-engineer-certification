#!/usr/bin/env python3
"""
Loader for the trained Academy models. This is what course labs import.

  load_tinygpt()            -> the base checkpoint, ready to generate
  load_tinygpt_with_lora()  -> the base + security/verification LoRA adapter
  generate(model, prompt, n)-> a text continuation (temperature 0 = deterministic)

Both loaders return a TinyGPT with its tokenizer attached as model.tok, so the
generate helper needs only the model. Everything is CPU, offline, no network.
"""
import os
import json
import torch

from model import (TinyGPT, CharTokenizer, inject_lora, load_lora_state_dict,
                   generate as _generate)

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "tinygpt.pt")
META = os.path.join(HERE, "tinygpt_meta.json")
ADAPTER = os.path.join(HERE, "lora_adapter.pt")


def _load_meta():
    with open(META) as f:
        return json.load(f)


def load_tinygpt():
    """Return the trained base TinyGPT (weights loaded, eval mode, tokenizer attached)."""
    meta = _load_meta()
    tok = CharTokenizer(list(meta["vocab"]))
    model = TinyGPT(meta["vocab_size"], meta["c"], meta["n_layer"], meta["n_block"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    model.tok = tok
    return model


def load_tinygpt_with_lora():
    """Return the base TinyGPT with the trained LoRA adapter attached."""
    meta = _load_meta()
    tok = CharTokenizer(list(meta["vocab"]))
    model = TinyGPT(meta["vocab_size"], meta["c"], meta["n_layer"], meta["n_block"])
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    # rebuild the same adapter shape, then load the trained adapter tensors.
    inject_lora(model, r=16, alpha=32)
    load_lora_state_dict(model, torch.load(ADAPTER, map_location="cpu"))
    model.eval()
    model.tok = tok
    return model


def generate(model, prompt, n=200, temperature=0.0):
    """Continue `prompt` for n characters. temperature 0 (default) is deterministic."""
    return _generate(model, model.tok, prompt, n=n, temperature=temperature, device="cpu")


if __name__ == "__main__":
    m = load_tinygpt()
    print("BASE:", generate(m, "a neural network is", n=120).replace("\n", " "))
    ml = load_tinygpt_with_lora()
    print("LoRA:", generate(ml, "a neural network is", n=120).replace("\n", " "))
