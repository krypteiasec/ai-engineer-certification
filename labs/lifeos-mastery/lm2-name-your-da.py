#!/usr/bin/env python3
"""
LAB LM2: Name your DA. The default is literally "PAI" until you change it.

Before you name it, your Digital Assistant answers to "PAI". That is a placeholder,
not a name. The doctrine is simple: everyone running LifeOS names their own DA, a
name, a voice, a personality, an identity. Miessler named his "Kai", his Digital
Assistant that will always be with him. The name lives in two places that must
agree: da.name in USER/Config/PAI_CONFIG.yaml, and the identity paragraph in
USER/DA_IDENTITY.md that gets @-imported every session. In this lab you set the
name on a SAMPLE config and prove the placeholder is gone and both files agree.
Everything here is invented for the lesson.

Run: python3 modules/academy-content/labs/lifeos-mastery/lm2-name-your-da.py
"""
import sys

# STEP 1: a fresh, un-named sample config. da.name is the default placeholder.
config = {
    "principal": {"name": "Sample User", "timezone": "UTC", "email": "sample@example.com"},
    "da": {"name": "PAI", "voice_id": "PLACEHOLDER"},
    "services": {"anthropic_key_env": "ANTHROPIC_API_KEY", "elevenlabs_key_env": "ELEVENLABS_API_KEY"},
    "pulse": {"port": 31337, "voice_enabled": True},
}
DEFAULT = "PAI"
print("STEP 1: fresh config, DA is unnamed")
print(f"  da.name    = {config['da']['name']!r}  (this is the placeholder)")
print(f"  is default : {config['da']['name'] == DEFAULT}")

# STEP 2: name it. The wizard / the /interview writes da.name, and the identity
# file's first-person paragraph must be rewritten to speak in that name.
def name_da(cfg, chosen, voice):
    cfg["da"]["name"] = chosen
    cfg["da"]["voice_id"] = voice
    identity_md = f"# {chosen} (DA Identity)\n\nI am {chosen}, your Digital Assistant. I will always be with you.\n"
    return identity_md

chosen_name = "Kai"    # the name a sample user picks; yours is your call
identity_md = name_da(config, chosen_name, "voice-sample-01")
print("")
print("STEP 2: set da.name and rewrite the identity paragraph")
print(f"  da.name       = {config['da']['name']!r}")
print(f"  DA_IDENTITY   -> {identity_md.splitlines()[2]!r}")

# STEP 3: invariants. The placeholder is gone, the config carries the real name,
# a voice is set, and DA_IDENTITY.md speaks in the SAME name (the two agree).
placeholder_gone = config["da"]["name"] != DEFAULT
name_set = config["da"]["name"] == chosen_name and config["da"]["voice_id"] != "PLACEHOLDER"
files_agree = (chosen_name in identity_md) and (chosen_name == config["da"]["name"])
print("")
print(f"STEP 3: placeholder replaced         : {placeholder_gone}")
print(f"        config carries name + voice  : {name_set}")
print(f"        config and DA_IDENTITY agree : {files_agree}")

ok = placeholder_gone and name_set and files_agree
print("")
print(f"DA NAMED (default PAI replaced, config and identity agree): {'YES' if ok else 'NO'}")
if not ok:
    sys.exit(1)
print("Your assistant has a name now, not a placeholder. Next: give it a personality.")
