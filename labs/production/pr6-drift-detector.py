#!/usr/bin/env python3
"""
LAB PR6: Drift and quality monitoring. Catching the input distribution shift.

A model that passed every eval on launch can still degrade in production without a
single line of code changing, because the WORLD changed. Users start asking about
things the system was never tuned for, the input distribution drifts away from the
data you evaluated on, and answer quality quietly falls. You cannot catch this with
unit tests. You catch it by monitoring the distribution of live traffic against the
baseline you trust, and raising a flag when the live inputs no longer look like the
baseline. This is data drift detection, and it is a core LLMOps monitor.

This lab embeds a baseline traffic sample, forms its centroid (the shape of
"normal"), and measures how far two live windows sit from that centroid using
cosine distance. The STABLE window is more of the same topic and sits close. The
SHIFTED window is a different topic entirely and sits far past the alert threshold.
Because the embeddings are deterministic, the flag is reproducible on any machine.

Run: python3 modules/academy-content/labs/production/pr6-drift-detector.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
from academy_llm import embed, cosine

# The baseline: the topic the system was evaluated and tuned on (sailing/harbor).
BASELINE = [
    "the sailboat crossed the harbor under a strong wind",
    "we trimmed the sails as the boat turned into the bay",
    "the harbor was full of boats waiting for the tide",
    "a steady wind carried the sloop across the open sea",
    "the captain steered the yacht past the rocky shore",
]
# A live window that is MORE of the same distribution: should look normal.
STABLE_WINDOW = [
    "the boat sailed across the bay with the wind behind it",
    "sailors trimmed the sails near the harbor at dawn",
    "a yacht crossed the sea toward the distant shore",
]
# A live window that has DRIFTED to a different topic: should trip the alert.
SHIFTED_WINDOW = [
    "the electron occupies a quantum orbital in the atom",
    "the algorithm computes gradients over the loss function",
    "inflation and interest rates moved the bond market today",
]


def centroid(texts):
    vecs = [embed(t) for t in texts]
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def mean_distance(texts, center):
    """Mean cosine DISTANCE (1 - similarity) of texts from a center vector."""
    return sum(1.0 - cosine(embed(t), center) for t in texts) / len(texts)


def main():
    base_center = centroid(BASELINE)

    # calibrate the alert threshold from the baseline's own spread + a margin.
    base_spread = mean_distance(BASELINE, base_center)
    threshold = base_spread + 0.25

    stable_dist = mean_distance(STABLE_WINDOW, base_center)
    shifted_dist = mean_distance(SHIFTED_WINDOW, base_center)

    print("STEP 1: measure each live window's distance from the baseline centroid")
    print("  baseline self-spread : %.3f" % base_spread)
    print("  alert threshold      : %.3f  (spread + 0.25 margin)" % threshold)
    print("  stable window  dist  : %.3f  ->  %s"
          % (stable_dist, "ALERT" if stable_dist > threshold else "ok"))
    print("  shifted window dist  : %.3f  ->  %s"
          % (shifted_dist, "ALERT" if shifted_dist > threshold else "ok"))

    stable_flagged = stable_dist > threshold
    shifted_flagged = shifted_dist > threshold

    print("")
    print("STEP 2: the detector must ignore normal traffic and flag the shift")
    print("  stable window flagged : %s (want NO)" % ("YES" if stable_flagged else "NO"))
    print("  shifted window flagged: %s (want YES)" % ("YES" if shifted_flagged else "NO"))

    # ---- proofs ----
    correct = (shifted_flagged is True) and (stable_flagged is False)
    ordered = shifted_dist > stable_dist
    ok = correct and ordered
    print("")
    print("DRIFT DETECTOR FLAGGED THE SHIFT: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Quality drift is now observable. Next: version and roll back safely.")


if __name__ == "__main__":
    main()
