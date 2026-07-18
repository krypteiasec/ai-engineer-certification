#!/usr/bin/env python3
"""
academy_llm: a tiny, deterministic, OFFLINE stand-in "LLM" for the teaching labs.

This is NOT a real model. It is a small rule-based machine that behaves ENOUGH
like a language model to make the mechanics visible and reproducible with zero
network, zero GPU, and zero dependencies (Python standard library only). The
Prompt Engineering, Agents, Evals, App Engineering, and Security labs all import
this one file so every lesson runs the same way on any machine, forever.

What it lets a lesson demonstrate for real:
  complete()    prompt in, text out. Few-shot examples in the prompt measurably
                change the answer. It can classify sentiment, extract JSON, and
                answer from provided context with real rules you can assert on.
  embed()       text in, a fixed 64-number vector out. Similar texts get vectors
                that point in a similar direction (higher cosine similarity), so
                a RAG retrieval lab can actually find the closest passage.
  chat()        the same completer wearing a messages[] coat, like a chat API.
  tool_route()  picks a tool by keyword, the core move of a ReAct agent.

Optional later upgrade: point complete()/embed() at a local LM Studio or Ollama
model behind these exact function names and the labs do not change. It stays a
mock so the labs never need a network or a GPU to teach.

Run the built-in self-test:  python3 modules/academy-content/labs/academy_llm.py
"""
import hashlib
import json
import math
import random
import re

EMBED_DIM = 64  # every embedding is exactly this many numbers, always


# --------------------------------------------------------------------------- #
# Tokenizing. A "token" here is just a lowercased word. Real models use subword
# tokens, but words are enough to teach retrieval and pattern matching.
# --------------------------------------------------------------------------- #
def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


# --------------------------------------------------------------------------- #
# embed(): turn text into a deterministic 64-dimensional vector by HASHING each
# token into a couple of dimensions. The same word always lands in the same
# slots, so two texts that share words share vector direction. That is the whole
# trick behind semantic search: similar text, similar vector.
# --------------------------------------------------------------------------- #
def embed(text):
    vec = [0.0] * EMBED_DIM
    for tok in _tokens(text):
        # hashlib gives us a stable, well-spread fingerprint of the word.
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # Use the fingerprint to choose which dimensions this word pushes on,
        # and in which direction (+ or -). Two dimensions per word adds a little
        # richness so different words collide less often.
        idx1 = int.from_bytes(h[0:4], "big") % EMBED_DIM
        idx2 = int.from_bytes(h[4:8], "big") % EMBED_DIM
        sign = 1.0 if (h[8] & 1) else -1.0
        vec[idx1] += sign
        vec[idx2] += sign * 0.5
    # L2-normalize so every vector has length 1. Then cosine similarity is just
    # the dot product, and it only measures DIRECTION, not how long the text is.
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec  # empty / all-punctuation text has no direction; leave zeros
    return [x / norm for x in vec]


def cosine(a, b):
    """Cosine similarity of two vectors: 1.0 is identical direction, 0.0 is
    unrelated, -1.0 is opposite. This is how a RAG lab ranks passages."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Reading a prompt. A real model reads free text; our mock reads a few simple
# shapes a lesson can rely on. We pull out any few-shot demonstrations and the
# final "query" the prompt is really asking us to answer.
# --------------------------------------------------------------------------- #
_SENTIMENT_POS = {"love", "great", "good", "excellent", "amazing", "happy",
                  "wonderful", "best", "fantastic", "like", "awesome", "nice",
                  "perfect", "brilliant", "enjoyed", "recommend"}
_SENTIMENT_NEG = {"hate", "terrible", "bad", "awful", "worst", "sad", "horrible",
                  "broken", "disappointing", "poor", "ugly", "useless", "angry",
                  "slow", "boring", "waste", "buggy"}


def _demonstrations(prompt):
    """Find few-shot examples written as `Input: X` / `Output: Y` pairs, or as
    `X -> Y` / `X => Y` lines. Returns a list of (input, output) pairs. An empty
    trailing output (the thing we are being asked to complete) is skipped."""
    pairs = []
    # Shape A: Input:/Output: line pairs.
    ins = re.findall(r"(?im)^\s*input\s*:\s*(.+?)\s*$", prompt)
    outs = re.findall(r"(?im)^\s*output\s*:\s*(.*?)\s*$", prompt)
    for i in range(min(len(ins), len(outs))):
        if outs[i] != "":  # a blank output is the query, not a demonstration
            pairs.append((ins[i], outs[i]))
    # Shape B: arrow lines like `great -> POS`.
    for m in re.findall(r"(?im)^\s*(.+?)\s*(?:->|=>)\s*(.+?)\s*$", prompt):
        left, right = m
        if right and not left.lower().startswith(("input", "output")):
            pairs.append((left, right))
    return pairs


def _query(prompt):
    """The text we actually have to act on: a trailing `Input: X` with an empty
    `Output:` if present, otherwise the last non-empty line of the prompt."""
    m = re.search(r"(?im)^\s*input\s*:\s*(.+?)\s*$\s*output\s*:\s*$", prompt)
    if m:
        return m.group(1)
    lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _match_demo_format(answer, demos):
    """Make a rule-based answer LOOK like the demonstrated outputs. If every
    example output is uppercase, the answer becomes uppercase too. This is how
    few-shot examples measurably steer the format of what comes back."""
    if not demos:
        return answer
    outs = [o for _, o in demos]
    if all(o.isupper() for o in outs if o.strip()):
        return answer.upper()
    if all(o.islower() for o in outs if o.strip()):
        return answer.lower()
    return answer


def _classify_sentiment(text):
    toks = set(_tokens(text))
    pos = len(toks & _SENTIMENT_POS)
    neg = len(toks & _SENTIMENT_NEG)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _extract_json(text):
    """Pull simple facts out of text into a small JSON object. Real extraction
    uses the model; here we use plain rules a lesson can predict and assert on."""
    obj = {}
    name = re.search(r"(?i)\bname\s+is\s+([A-Z][a-z]+)", text) or \
        re.search(r"(?i)\bi am\s+([A-Z][a-z]+)", text)
    if name:
        obj["name"] = name.group(1)
    num = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if num:
        val = num.group(1)
        obj["number"] = float(val) if "." in val else int(val)
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email:
        obj["email"] = email.group(0)
    return json.dumps(obj, sort_keys=True)


def _answer_from_context(prompt):
    """Question-answering the honest, mechanical way: split the given context
    into sentences and return the one that shares the most words with the
    question. Grounded answers only, no invention."""
    ctx = re.search(r"(?is)context\s*:\s*(.*?)\s*question\s*:", prompt)
    q = re.search(r"(?is)question\s*:\s*(.*?)\s*$", prompt)
    if not ctx or not q:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", ctx.group(1).strip())
    q_toks = set(_tokens(q.group(1)))
    best, best_score = "", -1
    for s in sentences:
        score = len(set(_tokens(s)) & q_toks)
        if score > best_score:
            best, best_score = s.strip(), score
    return best


def complete(prompt, temperature=0.0, seed=0):
    """The core surface: text in, text out, and deterministic given
    (prompt, temperature, seed).

    - Few-shot examples in the prompt change the output. If a demonstrated input
      matches the query exactly, we return its demonstrated output (recall). We
      also copy the demonstrated FORMAT (for example, uppercase labels).
    - It follows simple instructions for real: classify sentiment, extract JSON,
      or answer a question from a provided context.
    - temperature == 0.0 is greedy and stable (same input, same output, every
      time). temperature > 0.0 adds seeded pseudo-random variation via
      random.Random(seed), so the same seed always reproduces the same output.
    """
    demos = _demonstrations(prompt)
    query = _query(prompt)
    low = prompt.lower()

    # Few-shot RECALL: an example whose input matches the query wins outright.
    for din, dout in demos:
        if din.strip().lower() == query.strip().lower():
            return dout

    # Instruction following. Each branch is a real rule a lesson can assert on.
    if "sentiment" in low:
        answer = _classify_sentiment(query)
        return _match_demo_format(answer, demos)
    if "json" in low or "extract" in low:
        return _extract_json(query if query else prompt)
    if "context:" in low and "question:" in low:
        return _answer_from_context(prompt)

    # Fallback: a small "generation" surface so temperature has something to do.
    # Greedy (temperature 0) echoes the strongest words of the query in order.
    # With temperature > 0 we shuffle word choice using the seeded RNG, so the
    # output varies but stays reproducible for a fixed seed.
    words = _tokens(query if query else prompt)
    if not words:
        return ""
    vocab = sorted(set(words), key=lambda w: (-words.count(w), w))
    take = min(5, len(vocab))
    if temperature <= 0.0:
        chosen = vocab[:take]  # greedy: the most frequent words, stable order
    else:
        rng = random.Random(seed)
        pool = list(vocab)
        # Higher temperature reaches deeper into the (less likely) word pool.
        reach = min(len(pool), max(take, int(round(temperature * len(pool)))))
        chosen = rng.sample(pool[:reach], take)
    return " ".join(chosen)


def chat(messages):
    """A chat-style wrapper, like the messages API. Flattens roles into one
    prompt and calls complete(). The last user turn is treated as the query."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append("%s: %s" % (role, content))
    # Put the final user message last so _query() picks it up as the ask.
    return complete("\n".join(parts))


_TOOL_KEYWORDS = {
    "calculator": {"calculate", "compute", "sum", "multiply", "plus", "minus",
                   "math", "average", "percent", "+", "-", "*", "/"},
    "search": {"search", "find", "look up", "lookup", "who", "what", "google",
               "web", "latest", "news"},
    "weather": {"weather", "temperature", "forecast", "rain", "snow", "sunny",
                "hot", "cold", "humidity"},
    "calendar": {"schedule", "meeting", "calendar", "appointment", "remind",
                 "event", "tomorrow", "book"},
}


def tool_route(prompt, tools):
    """Pick ONE tool from `tools` by keyword, or return None if nothing fits.
    This is the routing decision at the heart of a ReAct agent: read the ask,
    choose the tool. Deterministic and easy to assert on in the Agents lab."""
    low = prompt.lower()
    best, best_score = None, 0
    for tool in tools:
        score = 0
        if tool.lower() in low:  # naming the tool directly is the strongest hint
            score += 2
        for kw in _TOOL_KEYWORDS.get(tool.lower(), set()):
            if kw in low:
                score += 1
        if score > best_score:
            best, best_score = tool, score
    return best if best_score > 0 else None


# --------------------------------------------------------------------------- #
# Self-test. Proves the invariants every lab relies on, prints ACADEMY_LLM OK,
# and exits 0. If any invariant breaks, it raises and exits non-zero.
# --------------------------------------------------------------------------- #
def _selftest():
    # 1. Few-shot demonstrably changes the output. The demonstrations show the
    #    label written in uppercase, so the answer comes back uppercased too.
    base = complete("Classify the sentiment of this review: I love this")
    shots = complete(
        "great -> POS\n"
        "awful -> NEG\n"
        "Classify the sentiment of this review: I love this"
    )
    assert base == "positive", "base sentiment should be positive, got %r" % base
    assert shots == "POSITIVE", "few-shot should uppercase the label, got %r" % shots
    assert base != shots, "few-shot examples must change the output"

    # 2. temperature == 0 is stable across two calls; a seed reproduces itself.
    p = "Complete this: the quick brown fox jumps over the lazy fox"
    assert complete(p, 0.0) == complete(p, 0.0), "temperature 0 must be stable"
    assert complete(p, 0.9, seed=7) == complete(p, 0.9, seed=7), "same seed must reproduce"

    # 3. Real instruction following a lesson can assert on.
    assert complete("Classify the sentiment: this is terrible and broken") == "negative"
    j = json.loads(complete("Extract JSON: my name is Ada and the number is 42"))
    assert j.get("name") == "Ada" and j.get("number") == 42, "JSON extraction failed: %r" % j
    ctx = ("Context: The moon orbits the earth. The sun is a star. "
           "Question: What orbits the earth?")
    assert "moon orbits" in complete(ctx).lower(), "context QA failed"

    # 4. Embeddings: fixed length, and similar texts are cosine-closer than
    #    dissimilar ones (the property RAG retrieval needs).
    a = embed("the cat sat on the warm mat")
    b = embed("a cat sat on a warm mat")          # near-identical wording
    c = embed("quantum physics electron orbital") # unrelated topic
    assert len(a) == EMBED_DIM, "embedding length must be %d" % EMBED_DIM
    assert cosine(a, b) > cosine(a, c), "similar texts must be cosine-closer"
    assert abs(cosine(a, a) - 1.0) < 1e-9, "a text is identical to itself"

    # 5. chat() and tool_route() do the obvious right thing.
    reply = chat([{"role": "system", "content": "You classify sentiment."},
                  {"role": "user", "content": "sentiment: I love this product"}])
    assert reply == "positive", "chat() sentiment failed: %r" % reply
    assert tool_route("what is the weather in Paris", ["search", "weather", "calculator"]) == "weather"
    assert tool_route("please calculate 2 plus 2", ["search", "weather", "calculator"]) == "calculator"
    assert tool_route("hello there", ["search", "weather"]) is None

    print("ACADEMY_LLM OK")


if __name__ == "__main__":
    _selftest()
