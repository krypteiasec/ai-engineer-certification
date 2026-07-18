#!/usr/bin/env python3
"""
LAB IP4: RAG vs fine-tune vs prompt, the decision drill.

The theory round rarely asks you to recite definitions. It asks for judgment:
"Here is a scenario. Would you use RAG, fine-tuning, or just prompting, and
why?" Reciting what each one IS is a failure signal. Knowing WHEN to reach for
each is the senior answer. The rule of thumb the strong candidates give:
  - PROMPT      when the task is achievable with instructions and a few examples
                on a capable base model. Cheapest, fastest, try it first.
  - RAG         when the task needs knowledge the model was not trained on,
                especially knowledge that is fresh, private, or changes often.
                You retrieve facts at query time instead of baking them in.
  - FINE-TUNE   when the task needs a new BEHAVIOR, format, tone, or narrow
                skill (not new facts), and you have enough labeled examples to
                teach it. You change the model's weights, not its context.

You encode that judgment as a decision function and PROVE it routes a table of
real scenarios the way an interviewer would expect.

Run: python3 modules/academy-content/labs/interview-prep/ip4-rag-vs-finetune-decision.py
"""
import sys


def decide(scenario):
    """Return 'prompt', 'rag', or 'finetune' for a scenario described by flags.

    Priority order matters and reflects real practice: reach for the cheapest
    tool that actually fits. Knowledge needs point to RAG; behavior/style needs
    with training data point to fine-tuning; everything else is prompting."""
    needs_fresh_or_private_facts = scenario.get("needs_fresh_or_private_facts", False)
    needs_new_behavior_or_format = scenario.get("needs_new_behavior_or_format", False)
    labeled_examples = scenario.get("labeled_examples", 0)
    simple_instruction_task = scenario.get("simple_instruction_task", False)

    # 1. Fresh, private, or changing FACTS is the classic RAG signal. You cannot
    #    fine-tune knowledge that changes daily, and prompting cannot hold a
    #    whole corpus, so you retrieve.
    if needs_fresh_or_private_facts:
        return "rag"

    # 2. A new BEHAVIOR/format/style with enough labeled data is the fine-tune
    #    signal. Fine-tuning teaches how to act, not what is true today.
    if needs_new_behavior_or_format and labeled_examples >= 500:
        return "finetune"

    # 3. A behavior need with too little data still cannot fine-tune well, so
    #    fall back to prompting with examples rather than overfit a tiny set.
    if needs_new_behavior_or_format and labeled_examples < 500:
        return "prompt"

    # 4. Anything a clear instruction and a few examples can do: just prompt.
    if simple_instruction_task:
        return "prompt"

    return "prompt"  # default to the cheapest tool


# Each row: a scenario an interviewer would actually pose, and the answer that
# gets you the offer.
cases = [
    ({"needs_fresh_or_private_facts": True,
      "simple_instruction_task": False},
     "rag", "Answer questions over the company's private, frequently-updated docs"),

    ({"needs_new_behavior_or_format": True, "labeled_examples": 5000},
     "finetune", "Force a strict house JSON style, 5000 labeled examples on hand"),

    ({"needs_new_behavior_or_format": True, "labeled_examples": 40},
     "prompt", "Want a specific tone but only 40 examples, too few to fine-tune"),

    ({"simple_instruction_task": True},
     "prompt", "Summarize a pasted article into three bullets"),

    ({"needs_fresh_or_private_facts": True,
      "needs_new_behavior_or_format": True, "labeled_examples": 9000},
     "rag", "Needs both fresh facts AND style: facts win, retrieve then shape"),
]

print("STEP 1: route each scenario")
all_ok = True
for scenario, expected, desc in cases:
    got = decide(scenario)
    ok = (got == expected)
    all_ok = all_ok and ok
    mark = "ok " if ok else "BAD"
    print(f"  [{mark}] {got:8s} (want {expected:8s})  {desc}")

print("")
print(f"STEP 2: every scenario routed as an interviewer expects : {all_ok}")

print("")
print(f"RAG VS FINETUNE VS PROMPT ROUTING MATCHES EXPECTED: {'YES' if all_ok else 'NO'}")
if not all_ok:
    sys.exit(1)
print("Judgment beats recitation. Next round: LLM system design.")
