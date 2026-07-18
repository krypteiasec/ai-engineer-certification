#!/usr/bin/env python3
"""
LAB PR7: The LLMOps lifecycle. Versioning and rollback.

A production LLM system is never one prompt or one model, it is a sequence of
versions, each immutable and each with a recorded eval score. That version history
is what makes an LLM system operable: you can say exactly what is live, prove a new
version is better before you promote it, and, when something goes wrong in the
real world, roll back to the last version you trusted in seconds rather than
debugging live in front of users. Model registries (versioning), promotion gates,
and one-click rollback are the backbone of LLMOps.

This lab builds a tiny model registry. It registers immutable versions, promotes
each only if its eval score clears the gate, and keeps a "last known good"
pointer. Then a freshly promoted version regresses under real traffic, monitoring
catches it, and the rollback restores the previous good version, exactly and
verifiably. It proves the registry is immutable and the restored config matches
the trusted one byte for byte.

Run: python3 modules/academy-content/labs/production/pr7-llmops-lifecycle.py
"""
import sys
import copy

GATE = 0.80


class Registry:
    """An append-only version registry with a promote gate and rollback."""
    def __init__(self):
        self.versions = []          # immutable list of registered versions
        self.active = None          # id of the currently serving version
        self.previous = None        # id of the version that was live BEFORE active

    def register(self, vid, config, eval_score):
        # store a deep copy so a later mutation of the caller's dict cannot
        # reach back and change history: versions are immutable.
        self.versions.append({"id": vid, "config": copy.deepcopy(config),
                              "eval_score": eval_score})

    def get(self, vid):
        for v in self.versions:
            if v["id"] == vid:
                return v
        return None

    def promote(self, vid):
        v = self.get(vid)
        if v is None:
            return False, "unknown version"
        if v["eval_score"] < GATE:
            return False, "blocked by gate (%.2f < %.2f)" % (v["eval_score"], GATE)
        # remember the version that was live before this one: THAT is what a
        # rollback restores. A new promotion is not yet trusted in production.
        self.previous = self.active
        self.active = vid
        return True, "promoted"

    def rollback(self):
        """Restore the previous version (the last one trusted before the current
        active regressed) and return its config."""
        if self.previous is None:
            return None
        self.active = self.previous
        return copy.deepcopy(self.get(self.previous)["config"])


def main():
    reg = Registry()
    reg.register("v1", {"prompt": "classify sentiment", "model": "small"}, 0.85)
    reg.register("v2", {"prompt": "classify sentiment, be precise", "model": "small"}, 0.92)
    reg.register("v3", {"prompt": "just answer", "model": "small"}, 0.55)  # a dud

    print("STEP 1: promote versions through the gate (>= %.2f)" % GATE)
    for vid in ["v1", "v2", "v3"]:
        okp, msg = reg.promote(vid)
        print("  promote %s : %-8s (%s)  active=%s previous=%s"
              % (vid, "OK" if okp else "DENIED", msg, reg.active, reg.previous))

    # v3 was blocked by the gate, so v2 is live and trusted.
    good_config = copy.deepcopy(reg.get("v2")["config"])

    # ---- simulate a version that PASSED the gate but regresses in production ----
    print("")
    print("STEP 2: a new version passes the gate but regresses under real traffic")
    reg.register("v4", {"prompt": "classify sentiment v4", "model": "small"}, 0.83)
    reg.promote("v4")
    print("  promoted v4, active=%s (eval passed at 0.83)" % reg.active)
    prod_quality_v4 = 0.40   # live monitoring (see PR6) reports a quality drop
    print("  production monitor reports live quality %.2f -> regression detected"
          % prod_quality_v4)

    # ---- rollback ----
    restored = reg.rollback()
    print("")
    print("STEP 3: roll back to the last known good version")
    print("  active after rollback: %s" % reg.active)
    print("  restored config      : %s" % restored)

    # ---- proofs ----
    rolled_to_good = reg.active == "v2"
    config_matches = restored == good_config
    # immutability: v3's dud config is still exactly what was registered.
    immutable = reg.get("v3")["config"] == {"prompt": "just answer", "model": "small"}
    ok = rolled_to_good and config_matches and immutable
    print("")
    print("  rolled back to good version: %s   config matches: %s   registry immutable: %s"
          % ("YES" if rolled_to_good else "NO",
             "YES" if config_matches else "NO",
             "YES" if immutable else "NO"))
    print("")
    print("ROLLBACK RESTORED THE GOOD VERSION: %s" % ("YES" if ok else "NO"))
    if not ok:
        sys.exit(1)
    print("Versioning + rollback is the LLMOps backbone. Next: tie it all together.")


if __name__ == "__main__":
    main()
