#!/usr/bin/env python3
"""
LAB MM4: Speech, from a waveform to a decision.

Audio reaches a model as a WAVEFORM: a long list of numbers, the air-pressure
value sampled thousands of times a second. Raw samples are too many and too low
level to reason over, so every speech system does the same first move: chop the
waveform into short overlapping FRAMES and compute a feature per frame (energy,
pitch, or a spectrum). Those features are what a recognizer actually reads. This
lab synthesizes two tiny "spoken tokens" as tones, a low-pitch one and a
high-pitch one, frames each waveform, measures pitch with a zero-crossing rate,
and proves a dead-simple classifier reads the right token from the features. That
is speech-to-text in miniature: waveform in, frames, features, label out.

No microphone, no audio model, no network. The waveforms are generated numbers,
and the classifier is a threshold on a real signal feature you can assert on.

Run: python3 modules/academy-content/labs/multimodal/mm4-speech.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break

import numpy as np

SR = 800          # samples per second (tiny, just for the demo)
DUR = 0.5         # seconds per token
FRAME = 80        # samples per frame
HOP = 40          # step between frames


def synth(freq):
    """Synthesize a pure-tone waveform at `freq` Hz. A real spoken word is a
    messy mix of frequencies, but a single tone is enough to show the pipeline:
    the low tone stands for one token, the high tone for another."""
    n = int(SR * DUR)
    t = np.arange(n, dtype=np.float32) / SR
    return np.sin(2.0 * np.pi * freq * t).astype(np.float32)


def frames(wave):
    """Chop the waveform into overlapping frames. Every speech feature is
    computed per frame, so framing is always the first step."""
    out = []
    for start in range(0, len(wave) - FRAME + 1, HOP):
        out.append(wave[start:start + FRAME])
    return np.array(out, dtype=np.float32)


def zero_cross_rate(frame):
    """Zero-crossing rate: how often the signal changes sign inside a frame.
    A higher pitch crosses zero more often, so ZCR is a cheap pitch proxy."""
    signs = np.sign(frame)
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def feature(wave):
    """One number per waveform: the average zero-crossing rate over all frames.
    This is the acoustic feature the classifier reads."""
    fr = frames(wave)
    return float(np.mean([zero_cross_rate(f) for f in fr]))


# Two known tokens, taught by their pitch. "yes" = low tone, "no" = high tone.
LOW_HZ, HIGH_HZ = 40.0, 160.0
templates = {"yes": feature(synth(LOW_HZ)), "no": feature(synth(HIGH_HZ))}
THRESH = (templates["yes"] + templates["no"]) / 2.0


def classify(wave):
    """Read the waveform's pitch feature and decide the token. Below the
    midpoint is the low tone (yes), above it is the high tone (no)."""
    return "yes" if feature(wave) < THRESH else "no"


print("STEP 1: synthesize two spoken tokens as waveforms")
print(f"  sample rate     : {SR} Hz, {int(SR*DUR)} samples per token")
print(f"  'yes' tone      : {LOW_HZ:.0f} Hz   'no' tone: {HIGH_HZ:.0f} Hz")

print("")
print("STEP 2: frame each waveform and measure the pitch feature (ZCR)")
for name, f in templates.items():
    print(f"  token '{name}'    frames={len(frames(synth(LOW_HZ if name=='yes' else HIGH_HZ)))}  ZCR={f:.3f}")
print(f"  decision line   : ZCR = {THRESH:.3f} (below -> yes, above -> no)")

print("")
print("STEP 3: classify fresh test waveforms from their features")
# Build test clips (regenerated, not the templates) and read the label back.
tests = [("yes", synth(LOW_HZ)), ("no", synth(HIGH_HZ)),
         ("yes", synth(LOW_HZ * 1.2)), ("no", synth(HIGH_HZ * 0.9))]
correct = 0
for truth, wave in tests:
    pred = classify(wave)
    hit = pred == truth
    correct += hit
    print(f"  ZCR={feature(wave):.3f}  predicted='{pred}'  truth='{truth}'  ({'ok' if hit else 'MISS'})")

ok = bool(correct == len(tests))
print("")
print(f"SPEECH FEATURES CLASSIFIED THE WAVEFORM: {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("")
print("Waveform, frames, features, label: that is the spine of speech-to-text,")
print("and text-to-speech runs it in reverse. Next: retrieve across BOTH images")
print("and text at once, the multimodal RAG the applied job actually asks for.")
