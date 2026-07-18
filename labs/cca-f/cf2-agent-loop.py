#!/usr/bin/env python3
"""
LAB CF2: The agent loop and stop_reason (Domain 1, 27%).

Every API call returns a stop_reason. The loop runs until stop_reason is
"end_turn", executing a tool on every "tool_use". That is the only reliable
completion signal. This lab implements the loop as a small state machine and
proves two things the exam tests directly: the correct loop terminates exactly
when the model ends its turn, and the tempting anti-patterns (parsing the
assistant's text for a completion word, or using an iteration cap as the primary
stop) fail on the same trace.

Deterministic, offline, standard library only.
Run: python3 modules/academy-content/labs/cca-f/cf2-agent-loop.py
"""
import sys

# A scripted transcript: what the "model" returns on each call. Each turn has a
# stop_reason and some text. Two tool_use turns, then it ends the turn.
TRANSCRIPT = [
    {"stop_reason": "tool_use", "text": "I am done thinking, let me look that up"},
    {"stop_reason": "tool_use", "text": "done with the first tool, one more"},
    {"stop_reason": "end_turn", "text": "Here is your final answer"},
]

# The correct action for each stop_reason (from the study guide table).
ACTION = {
    "tool_use": "execute tool, append result, call again",
    "end_turn": "finished, show the user",
    "max_tokens": "truncated, raise limit or split",
    "stop_sequence": "handle per application logic",
}


def correct_loop(transcript):
    """Drive the loop the right way: terminate only on end_turn, run a tool on
    every tool_use. Returns (tools_run, final_text)."""
    tools_run = 0
    for turn in transcript:
        if turn["stop_reason"] == "end_turn":
            return tools_run, turn["text"]
        if turn["stop_reason"] == "tool_use":
            tools_run += 1  # execute the tool, then the loop would call again
    raise RuntimeError("transcript ended without end_turn")


def text_parsing_antipattern(transcript):
    """WRONG: stop as soon as the assistant text contains a completion word.
    This fires on turn 1 ('I am done thinking'), ending the loop early and
    skipping both real tool calls."""
    for i, turn in enumerate(transcript):
        if "done" in turn["text"].lower():
            return i + 1  # number of turns before it (wrongly) stopped
    return len(transcript)


def iteration_cap_antipattern(transcript, cap=1):
    """WRONG: use an arbitrary iteration cap as the PRIMARY stop condition. With
    cap=1 it halts after one turn, before the model has ended its turn."""
    for i, turn in enumerate(transcript):
        if i + 1 >= cap:
            return i + 1
    return len(transcript)


tools_run, final = correct_loop(TRANSCRIPT)
print("STEP 1: the correct loop (terminate only on end_turn)")
for i, turn in enumerate(TRANSCRIPT):
    print(f"  call {i+1}: stop_reason={turn['stop_reason']:<12} -> {ACTION[turn['stop_reason']]}")
print(f"  tools executed: {tools_run}   final text: {final!r}")

early_stop = text_parsing_antipattern(TRANSCRIPT)
cap_stop = iteration_cap_antipattern(TRANSCRIPT, cap=1)
print("")
print("STEP 2: the anti-patterns the exam plants as distractors")
print(f"  parse-text-for-'done'  stopped after turn {early_stop} (should be 3)")
print(f"  iteration-cap(=1)      stopped after turn {cap_stop} (should be 3)")

# Invariants.
loop_ok = (tools_run == 2 and final == "Here is your final answer")
text_bad = early_stop < len(TRANSCRIPT)     # the text anti-pattern stops too early
cap_bad = cap_stop < len(TRANSCRIPT)        # the cap anti-pattern stops too early
end_turn_is_signal = TRANSCRIPT[-1]["stop_reason"] == "end_turn"

ok = loop_ok and text_bad and cap_bad and end_turn_is_signal
print("")
print(f"  correct loop ran both tools and ended cleanly : {loop_ok}")
print(f"  text-parsing stop fired too early (is wrong)  : {text_bad}")
print(f"  iteration-cap stop fired too early (is wrong) : {cap_bad}")
print("")
print(f"AGENT LOOP TERMINATES ONLY ON end_turn: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("stop_reason == end_turn is the only reliable completion signal. Next: orchestration and hooks.")
