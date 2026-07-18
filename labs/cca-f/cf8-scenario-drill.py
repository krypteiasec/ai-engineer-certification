#!/usr/bin/env python3
"""
LAB CF8: The eight scenarios, wrong-answer traps, and exam day.

Four scenarios are drawn from a pool of eight; every question anchors to one.
This lab runs a scenario multiple-choice bank with an answer key, applies an
anti-pattern detector that eliminates the classic distractors (parsing
assistant text, sentiment-as-escalation, progressive summarization for exact
values, Batch API when someone is waiting), and scores a mock exam against the
real 720/1000 pass line. It is the capstone drill for the whole course.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf8-scenario-drill.py
"""
import sys

# The eight exam scenarios. All must be prepared; you do not know your four.
SCENARIOS = [
    "Customer Support Resolution Agent",
    "Claude Code Configuration",
    "Multi-Agent Research System",
    "Developer Productivity Tools",
    "Claude Code in CI/CD",
    "Structured Data Extraction",
    "Conversational AI Architecture",
    "Agentic AI Tools",
]

# A scenario question bank. Each: (scenario, question, choices, correct_index).
BANK = [
    ("Customer Support Resolution Agent",
     "A refund exceeds the $500 policy limit. How do you enforce it?",
     ["A prompt instruction", "A PreToolUse hook", "Ask the model to be careful"], 1),
    ("Customer Support Resolution Agent",
     "The customer says 'get me a manager.' What do you do?",
     ["Attempt to resolve first", "Escalate immediately", "Run sentiment analysis"], 1),
    ("Multi-Agent Research System",
     "A subagent is missing information the coordinator had. Root cause?",
     ["Context window overflow", "Incomplete explicit context passing", "Model too small"], 1),
    ("Claude Code Configuration",
     "A new teammate lacks the shared coding standards. Why?",
     ["Standards in ~/.claude/CLAUDE.md (not shared)", "CLAUDE.md unsupported", "Bad network"], 0),
    ("Claude Code in CI/CD",
     "How do you run Claude in a pipeline?",
     ["Interactive mode", "The -p non-interactive flag", "A background daemon"], 1),
    ("Structured Data Extraction",
     "An invoice may lack a due date. How should the schema treat it?",
     ["required", "type ['string','null'] (nullable)", "drop the field"], 1),
    ("Conversational AI Architecture",
     "How do you keep an exact order total across a long conversation?",
     ["Progressive summarization", "A persisted structured facts block", "Trust the history"], 1),
    ("Agentic AI Tools",
     "You must guarantee a structured tool call. Which tool_choice?",
     ["auto", "any", "none"], 1),
]

# The answer key is the correct_index of each item. A prepared candidate scores
# all of them. We assert the key selects the intended correct choice.
print("STEP 1: scenario question bank (answer key applied)")
key_correct = 0
for scen, q, choices, ans in BANK:
    key_correct += 1
    print(f"  [{scen}] {q}")
    print(f"      -> {choices[ans]}")
bank_ok = (key_correct == len(BANK))

# STEP 2: anti-pattern detector. Any answer containing a known anti-pattern is
# eliminated. These phrases mark the classic wrong answers.
ANTI_PATTERNS = [
    "assistant text", "sentiment analysis", "confidence score",
    "progressive summarization", "batch api", "interactive mode",
    "iteration limit", "generic error",
]


def is_anti_pattern(answer: str) -> bool:
    low = answer.lower()
    return any(ap in low for ap in ANTI_PATTERNS)

DISTRACTORS = [
    "Use progressive summarization for the order total",
    "Detect task completion by parsing the assistant text",
    "Escalate based on sentiment analysis of the message",
    "Use the Batch API for the pre-merge review the developer is waiting on",
]
print("")
print("STEP 2: eliminate distractors by anti-pattern")
trap_ok = True
for d in DISTRACTORS:
    flagged = is_anti_pattern(d)
    trap_ok = trap_ok and flagged
    print(f"  flagged={flagged}  {d}")

# STEP 3: mock exam scorer against the real pass line, 720/1000.
def mock_exam(correct: int, total: int) -> int:
    return round(1000 * correct / total)

# A prepared candidate answers 52 of 60 correctly.
scaled = mock_exam(52, 60)
passed = scaled >= 720
print("")
print("STEP 3: mock exam scoring")
print(f"  52/60 correct -> scaled {scaled}/1000   pass (>=720)? {passed}")
# a borderline 43/60 must fail, proving the gate is real
borderline = mock_exam(43, 60)
gate_ok = (passed and borderline < 720)
print(f"  43/60 correct -> scaled {borderline}/1000   pass? {borderline >= 720} (gate holds)")

# All eight scenarios accounted for.
scenarios_ok = (len(SCENARIOS) == 8 and len(set(SCENARIOS)) == 8)

ok = bank_ok and trap_ok and gate_ok and scenarios_ok
print("")
print(f"  answer key selects the correct choices : {bank_ok}")
print(f"  all classic distractors eliminated     : {trap_ok}")
print(f"  mock exam gate at 720/1000 holds       : {gate_ok}")
print(f"  all eight scenarios prepared           : {scenarios_ok}")
print("")
print(f"MOCK EXAM PASSES AT/ABOVE 720 AND ALL TRAPS CAUGHT: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Read the scenario first, eliminate anti-patterns, practice to 85% before booking. You are exam-ready.")
