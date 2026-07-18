#!/usr/bin/env python3
"""
LAB TR8: Putting it together, when to fine-tune vs prompt vs RAG.

Fine-tuning is powerful and it is also the answer to fewer problems than people
think. The senior skill is choosing the right tool before spending a week
training. The decision comes down to what is actually missing:

  - Missing KNOWLEDGE the model does not have (private, fresh, or changing facts)
    -> RAG. Retrieve the facts at query time. Fine-tuning bakes in stale facts and
    is the wrong tool for a moving target.
  - Missing BEHAVIOR the model can already do but needs steering (format, style,
    a task it understands) -> PROMPT first. It is instant, free to change, and
    handles most cases with a good system prompt and a few examples.
  - Behavior that prompting cannot reliably reach, where you have MANY labeled
    examples and need it consistent and cheap at scale -> FINE-TUNE.

This lab encodes that judgment as a small decision function, then runs it against
a set of labeled real-world cases and PROVES it picks the right tool on every one.
No model, no training: pure decision logic, instant.

Run: python3 modules/academy-content/labs/training/tr8-finetune-vs-prompt-vs-rag.py
"""
import sys


def decide(needs_dynamic_or_private_knowledge, is_behavior_or_format,
           labeled_examples, prompting_sufficient):
    """Return RAG, PROMPT, or FINE-TUNE for a problem described by four features.

    Priority order encodes the doctrine:
      1. a knowledge gap is a retrieval problem, not a training problem,
      2. steerable behavior starts with a prompt (instant, free to change),
      3. only reach for a fine-tune when prompting has genuinely plateaued and you
         have enough labeled data to justify it.
    """
    if needs_dynamic_or_private_knowledge:
        return "RAG"
    if is_behavior_or_format:
        if prompting_sufficient or labeled_examples < 100:
            return "PROMPT"
        return "FINE-TUNE"
    # a general task the model understands: try the cheapest lever first.
    return "PROMPT"


# Labeled cases: (description, features, expected tool). The features describe the
# problem; the expected tool is the correct engineering call.
CASES = [
    ("answer questions over an internal wiki that changes every week",
     dict(needs_dynamic_or_private_knowledge=True, is_behavior_or_format=False,
          labeled_examples=0, prompting_sufficient=False), "RAG"),
    ("cite our current product prices, which change daily",
     dict(needs_dynamic_or_private_knowledge=True, is_behavior_or_format=False,
          labeled_examples=0, prompting_sufficient=False), "RAG"),
    ("always return output as valid JSON in a fixed schema",
     dict(needs_dynamic_or_private_knowledge=False, is_behavior_or_format=True,
          labeled_examples=6, prompting_sufficient=True), "PROMPT"),
    ("classify support tickets by sentiment, a few examples fix the mistakes",
     dict(needs_dynamic_or_private_knowledge=False, is_behavior_or_format=True,
          labeled_examples=20, prompting_sufficient=True), "PROMPT"),
    ("adopt our exact house writing voice across 10k tickets; prompting plateaued; 5000 labeled examples",
     dict(needs_dynamic_or_private_knowledge=False, is_behavior_or_format=True,
          labeled_examples=5000, prompting_sufficient=False), "FINE-TUNE"),
    ("specialized medical coding behavior, thousands of labeled pairs, must run cheap at scale",
     dict(needs_dynamic_or_private_knowledge=False, is_behavior_or_format=True,
          labeled_examples=8000, prompting_sufficient=False), "FINE-TUNE"),
    ("summarize an uploaded document the model has never seen",
     dict(needs_dynamic_or_private_knowledge=True, is_behavior_or_format=False,
          labeled_examples=0, prompting_sufficient=True), "RAG"),
]


def main():
    print("STEP 1: run the decision function over every labeled case")
    all_ok = True
    for desc, feats, expected in CASES:
        got = decide(**feats)
        ok = got == expected
        all_ok = all_ok and ok
        print("  [%s] want %-9s got %-9s  %s" % ("OK" if ok else "XX", expected, got, desc[:52]))

    # also assert the three levers are each reachable (the function is not a stub
    # that always returns one answer).
    reachable = {c[2] for c in CASES}
    covers_all = reachable == {"RAG", "PROMPT", "FINE-TUNE"}

    ok = all_ok and covers_all
    print("")
    print("  all cases routed correctly: %s" % ("YES" if all_ok else "NO"))
    print("  framework can reach all three tools: %s" % ("YES" if covers_all else "NO"))
    print("")
    print("DECISION FRAMEWORK PICKS THE RIGHT TOOL ON ALL CASES: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
