#!/usr/bin/env python3
"""
Deterministic, offline corpus for training the Academy tinyGPT.

No network, no downloads. We author a pool of clean, factual sentences about
programming and AI (the domain the whole Academy is about), then deterministically
compose them into paragraphs with a fixed seed. The result is a few hundred KB of
real English prose with a small, repeating vocabulary, which is exactly what a
tiny char-level model can actually learn to reproduce coherently.

Running this file writes two files to disk (committed artifacts):
  corpus.txt           -> the base-model training text (broad AI/programming prose)
  corpus_finetune.txt  -> the downstream LoRA task: a narrow SECURITY/VERIFICATION
                          subdomain, the classic "adapt a general model to one
                          domain" use case for LoRA.

The LoRA effect is visible and assertable: the base model, prompted about neural
networks, keeps talking about layers and tools; the LoRA-adapted model steers the
SAME frozen weights toward security and verification prose (check inputs, trust
nothing, verify by running). Every character in the fine-tune set is already in
the base vocabulary, so the adapter works WITH the frozen output head and the text
stays coherent.

Run: python3 dataset.py
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.txt")
FINETUNE_PATH = os.path.join(HERE, "corpus_finetune.txt")

SEED = 1234
TARGET_BYTES = 260_000
FINETUNE_BYTES = 90_000

# A pool of clean, self-contained factual sentences. Small closed vocabulary,
# heavy word reuse, no em dashes. This is what lets a char model learn words.
SENTENCES = [
    "a program is a list of instructions that a computer runs in order.",
    "a variable is a name that stores a value the program can read and change.",
    "a function takes an input, does some work, and returns an output.",
    "a loop repeats the same block of code until a condition is met.",
    "a list holds many values in order, and each value has an index.",
    "python is a popular language for writing clear and simple code.",
    "code is easier to read when the names describe what the values mean.",
    "a bug is a mistake in the code that makes the program behave wrong.",
    "we fix a bug by finding the cause and changing the code that made it.",
    "a test checks that a function returns the value we expect.",
    "when the test passes, we trust that the code does the right thing.",
    "a model learns patterns from data instead of following fixed rules.",
    "a neural network is a stack of layers that transform numbers into numbers.",
    "each layer has weights, and training slowly nudges the weights to be better.",
    "training measures how wrong the model is, then makes it a little less wrong.",
    "the loss is a number that goes down as the model learns the data.",
    "a language model predicts the next token from the tokens that came before.",
    "a token is a small piece of text, a character or a common group of letters.",
    "attention lets the model look back at earlier tokens that matter most.",
    "a transformer is a network built from attention and simple linear layers.",
    "gradient descent follows the slope of the loss down toward a better answer.",
    "the optimizer updates every weight a small step in the right direction.",
    "more data and more layers usually make a language model more capable.",
    "the same tiny machine, made much bigger, becomes a real language model.",
    "we split data into training and testing so we can measure real progress.",
    "an embedding turns a token into a vector of numbers the model can use.",
    "the output layer turns the final vector back into a score for each token.",
    "a good prompt gives the model clear context and a clear task to do.",
    "an agent is a model that can plan steps and use tools to reach a goal.",
    "a tool lets the model read a file, run a command, or search the web.",
    "we verify the result by running it, not by trusting that it looks right.",
    "clean code is short, clear, and does one thing at a time.",
    "reading code is a skill you build by reading a lot of good code.",
    "the best way to learn to build is to build a small thing end to end.",
    "a good engineer builds systems, ships real work, and learns every day.",
    "security means checking every input and trusting nothing by default.",
    "a cache stores a result so the program does not compute it twice.",
    "recursion is a function that calls itself on a smaller part of the problem.",
    "an array of weights is just a table of numbers the model can learn.",
    "the goal of training is a model that generalizes, not one that memorizes.",
]

# A couple of pangram-style lines so every letter appears in the base corpus,
# keeping all token embeddings alive.
PANGRAMS = [
    "the quick brown fox jumps over the lazy dog while five wizards vex.",
    "pack my box with five dozen liquor jugs, and quickly judge the vow.",
]

# The downstream LoRA subdomain: a narrow security/verification voice. All words
# use characters already present in the base vocabulary, so the frozen output
# head can render them and the adapted text stays coherent.
FINETUNE_SENTENCES = [
    "security means checking every input and trusting nothing by default.",
    "we verify the result by running it, never by guessing that it looks right.",
    "a test that passes is the only proof that the code really works.",
    "never trust data from outside without checking it first.",
    "verify every claim with a tool before you believe it is true.",
    "an attacker sends bad input, so we check the input before we use it.",
    "the safe habit is to trust nothing and to verify everything.",
    "we test the code, we run the code, and we confirm the output is correct.",
    "a secure program treats every input as dangerous until it is checked.",
    "checking the input stops the bug before it can ever run.",
    "we confirm that the model did the work by verifying the real output.",
    "trust is earned by proof, and proof means running the test and seeing it pass.",
]


def _compose(pool, target, rng):
    out, size = [], 0
    while size < target:
        para_len = rng.randint(3, 7)
        para = " ".join(rng.choice(pool) for _ in range(para_len))
        out.append(para)
        size += len(para) + 2
    return "\n\n".join(out) + "\n"


def main():
    rng = random.Random(SEED)
    text = _compose(SENTENCES + PANGRAMS, TARGET_BYTES, rng)
    fine = _compose(FINETUNE_SENTENCES, FINETUNE_BYTES, rng)
    with open(CORPUS_PATH, "w") as f:
        f.write(text)
    with open(FINETUNE_PATH, "w") as f:
        f.write(fine)
    print("wrote %s  (%d bytes)" % (CORPUS_PATH, len(text)))
    print("wrote %s  (%d bytes)" % (FINETUNE_PATH, len(fine)))
    # sanity: every fine-tune char must exist in the base vocabulary.
    missing = set(fine) - set(text)
    print("base vocab size: %d chars | fine-tune chars missing from base: %s"
          % (len(sorted(set(text))), sorted(missing) if missing else "none"))
    print("sample:\n" + text[:200])


if __name__ == "__main__":
    main()
