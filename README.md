<p align="center">
  <a href="https://krypteiasec.com/academy">
    <img src="assets/hero.jpg" alt="Krypteia AI Engineer Certification" width="100%">
  </a>
</p>

<h1 align="center">AI Engineer Certification</h1>

<p align="center">
  <b>Zero to hireable. Completely free.</b><br>
  19 courses &nbsp;·&nbsp; 151 chapters &nbsp;·&nbsp; 151 runnable labs &nbsp;·&nbsp; build a real LLM from scratch
</p>

<p align="center">
  <img src="https://img.shields.io/badge/price-free-7a1f1f?style=flat-square" alt="Free">
  <img src="https://img.shields.io/badge/license-MIT-5c1717?style=flat-square" alt="MIT">
  <img src="https://img.shields.io/badge/courses-19-b53d3d?style=flat-square" alt="19 courses">
  <img src="https://img.shields.io/badge/labs-151-b53d3d?style=flat-square" alt="151 labs">
  <img src="https://img.shields.io/badge/setup-none-5c1717?style=flat-square" alt="No setup">
</p>

<p align="center">
  <a href="https://krypteiasec.com/academy"><b>Start the certification →</b></a>
</p>

---

### ▶ Watch the intro

<video src="https://github.com/user-attachments/assets/92776c71-69fa-415c-846c-ed583088418e" controls muted width="100%"></video>

If the player does not load, [watch the intro here](https://github.com/user-attachments/assets/92776c71-69fa-415c-846c-ed583088418e).

---

Everyone says AI will take your job. This certification is built on the opposite bet: that the people who can *build* with AI become unstoppable. It takes you from Python and the math you need all the way to shipping AI agents and building a language model by hand, with a runnable lab for every single chapter. No paywall, no login, no catch.

The full interactive certification, prose teaching, narrated videos, quizzes, in-browser labs, and a free certificate, lives at **[krypteiasec.com/academy](https://krypteiasec.com/academy)**. This repository is the code layer: everything you clone, read, and run yourself.

## What is inside

```
labs/
  academy_llm.py        a tiny deterministic offline stand-in "LLM" the teaching labs import
  _models/              two REAL trained checkpoints + the training code
  foundations/          Course 0 labs
  <course>/             one folder of labs per course (prompt-engineering, rag, ...)
notebooks/              151 Jupyter notebooks, one per lab
courses.export.json     the machine-readable catalog the site consumes
COURSES.md              the human catalog: every course, chapter, and its lab
requirements.txt        torch, numpy, jupyter (only for the PyTorch + notebook labs)
```

## The curriculum

**Core track (Courses 0 to 13):** Python and the math you need, building a tiny LLM by hand, prompt engineering, RAG, agents and MCP, evaluation, application engineering, training and fine-tuning, transformer internals, production and LLMOps, safety and security, multimodal, interview prep, and a capstone portfolio.

**Applied track (Courses 14 to 18):** building agents with Claude Code, the Claude API and Agent SDK, certification prep, setting up a personal AI operating system, and a flagship mastery course.

Full detail in [`COURSES.md`](COURSES.md).

## Run it

Two ways to work through any lab, both first-class.

**Terminal.** Every lab is a standalone script that prints an invariant proving the concept and exits 0.

```bash
python3 labs/prompt-engineering/pe2-few-shot.py
python3 labs/rag/rag6-rag-pipeline.py
```

Most labs are pure standard library and need no install. The PyTorch labs and the trained models need `pip install -r requirements.txt`.

**Jupyter.** Each lab has a matching notebook, split into cells with the explanation above each step.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Or run **117 of the 151 labs with zero setup, right in your browser** at [krypteiasec.com/academy](https://krypteiasec.com/academy) (Python compiled to WebAssembly).

## The trained models (`labs/_models/`)

Two genuinely trained checkpoints you can load and generate from, not mocks:

- `tinygpt.pt`: a from-scratch character-level GPT (the same tiny architecture you build by hand in Course 1).
- `lora_adapter.pt`: a real LoRA fine-tune of that base model, with the base weights frozen.

```bash
python3 labs/_models/verify.py          # loads both, generates, prints MODELS OK
python3 labs/_models/train_tinygpt.py   # retrain the base from scratch (~15s on Apple Silicon)
```

## License

MIT, see [`LICENSE`](LICENSE). The curriculum and code are free to read, run, fork, and teach from.

<p align="center"><sub>Built by <a href="https://krypteiasec.com">Krypteia Sec</a> · an independent educational resource on AI and agentic engineering.</sub></p>
