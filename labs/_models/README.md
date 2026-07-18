# Academy trained models

Two genuinely trained checkpoints for the AI Engineering Academy, built from the
exact TinyGPT architecture students construct in Course 1 (labs 07, 08, 10).
Everything here is offline and torch only. No downloads, no network, no stubs.

## What each file is

| File | Role |
|------|------|
| `model.py` | The shared architecture. Same CausalAttention + Block + TinyGPT as lab 07, made config driven, plus the LoRA adapter layers and a char tokenizer. Imported by everything else. |
| `dataset.py` | Deterministic corpus generator. Writes `corpus.txt` (broad AI/programming prose) and `corpus_finetune.txt` (a narrow security/verification subdomain for the LoRA). Fixed seed, committed to disk. |
| `corpus.txt` | Base training text, about 260 KB of real prose over a 30 character vocab. |
| `corpus_finetune.txt` | The downstream LoRA task text, about 90 KB, security/verification voice. |
| `train_tinygpt.py` | Trains the TinyGPT from scratch on `corpus.txt`. The lab 08 loop (forward, loss, backward, step) scaled up to the full transformer. |
| `tinygpt.pt` | Trained base weights (state_dict), about 3.5 MB. |
| `tinygpt_meta.json` | Vocab string and config (width, layers, context) so the loader can rebuild the model. |
| `finetune_lora.py` | Real LoRA fine tune ON the trained TinyGPT. Freezes the base, trains only low rank A/B adapters on the attention query and value projections. |
| `lora_adapter.pt` | Trained LoRA adapter tensors only, about 134 KB. |
| `load.py` | The import surface for course labs: `load_tinygpt()`, `load_tinygpt_with_lora()`, `generate(model, prompt, n)`. |
| `verify.py` | Self test. Loads both checkpoints, asserts determinism and the LoRA subdomain shift, prints `MODELS OK`. |

## The model

Character level TinyGPT, single head causal attention, identical in shape to the
Course 1 labs, just scaled enough to learn:

- embedding width C = 128
- transformer blocks = 4
- context window = 128
- vocab = 30 characters
- about 848 K parameters

## The LoRA task

Classic LoRA use case: adapt a general model to one narrow domain. The base model,
prompted with "a neural network is", keeps talking about layers and tokens. The
LoRA adapted model steers the SAME frozen weights toward security and verification
prose (check inputs, trust nothing, verify by running). Only about 3.9% of the
parameters (the adapters) are trained; the base is frozen. The shift is visible on
a fixed prompt and asserted by counting subdomain keywords.

## How to retrain

```
python3 dataset.py          # regenerate the corpora (deterministic)
python3 train_tinygpt.py    # train the base -> tinygpt.pt + tinygpt_meta.json
python3 finetune_lora.py    # LoRA fine tune -> lora_adapter.pt
python3 verify.py           # self test, prints MODELS OK
```

## Expected runtime and sizes

On an Apple Silicon Mac (MPS), measured:

- `train_tinygpt.py`: about 16 seconds, 1500 steps, loss 3.5 down to 0.11
- `finetune_lora.py`: about 16 seconds, 2000 steps, loss 4.3 down to 0.09
- `verify.py`: a couple of seconds

Checkpoint sizes: `tinygpt.pt` about 3.5 MB, `lora_adapter.pt` about 134 KB.
CPU only works too, just slower.

## Using them in a lab

```python
from load import load_tinygpt, load_tinygpt_with_lora, generate

base = load_tinygpt()
print(generate(base, "a neural network is", n=120))

lora = load_tinygpt_with_lora()
print(generate(lora, "a neural network is", n=120))
```
