#!/usr/bin/env python3
"""
LAB LM3: The personality builder. A DA's character is quantified, not vibes.

DA_IDENTITY.md is where your assistant gets a self. The personality is not a mood
you hope for, it is twelve traits scored 0 to 100: enthusiasm, energy,
expressiveness, resilience, composure, optimism, warmth, formality, directness,
precision, curiosity, playfulness. LifeOS ships presets (efficient, friendly,
creative, mentor, worker) as starting points you then override. The "efficient"
preset, for example, runs directness 90, precision 95, warmth 40, playfulness 20.
In this lab you take a preset, apply a few overrides, validate every trait is in
range, and render an identity.md from a template with the placeholders filled.
Sample data only.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm3-personality-builder.py
"""
import sys

TRAITS = ["enthusiasm", "energy", "expressiveness", "resilience", "composure",
          "optimism", "warmth", "formality", "directness", "precision",
          "curiosity", "playfulness"]

# STEP 1: a preset is a full 12-trait baseline. Here is "efficient".
efficient = {
    "enthusiasm": 45, "energy": 55, "expressiveness": 40, "resilience": 80,
    "composure": 85, "optimism": 55, "warmth": 40, "formality": 60,
    "directness": 90, "precision": 95, "curiosity": 70, "playfulness": 20,
}
print("STEP 1: start from the 'efficient' preset")
print(f"  directness={efficient['directness']} precision={efficient['precision']} "
      f"warmth={efficient['warmth']} playfulness={efficient['playfulness']}")

# STEP 2: compose = preset + overrides. The user nudges a few traits to taste.
def compose(preset, overrides):
    persona = dict(preset)
    for k, v in overrides.items():
        persona[k] = v
    return persona

overrides = {"warmth": 55, "playfulness": 35, "curiosity": 85}
persona = compose(efficient, overrides)
print("")
print("STEP 2: apply overrides", overrides)
for k in overrides:
    print(f"  {k}: {efficient[k]} -> {persona[k]}")

# STEP 3: validate. Every one of the 12 traits must be present and in 0..100.
present = all(t in persona for t in TRAITS) and len(persona) == len(TRAITS)
in_range = all(0 <= persona[t] <= 100 for t in TRAITS)
overrides_applied = all(persona[k] == v for k, v in overrides.items())

# STEP 4: render identity.md from a template with placeholders filled.
TEMPLATE = "# {name} (DA Identity)\n\n## Personality\n{trait_lines}\n"
trait_lines = "\n".join(f"- {t}: {persona[t]}" for t in TRAITS)
identity_md = TEMPLATE.format(name="Aria", trait_lines=trait_lines)
rendered = ("{" not in identity_md) and all(str(persona[t]) in identity_md for t in TRAITS)
print("")
print("STEP 4: rendered identity.md (first 4 trait lines)")
for line in identity_md.splitlines()[3:7]:
    print("  " + line)

ok = present and in_range and overrides_applied and rendered
print("")
print(f"PERSONA BUILT (12 traits present and in range, overrides applied, identity rendered): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Your DA has a measurable character now. Next: TELOS, the reason it exists.")
