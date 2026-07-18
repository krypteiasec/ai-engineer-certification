# Krypteia AI Engineer Certification, companion code

The hands-on code for a complete, free, zero-to-hireable AI Engineering certification: nineteen courses, 151 chapters, one runnable Python lab plus a Jupyter notebook for every single chapter, and two real trained model checkpoints. Everything runs offline and deterministically, so a lab either proves its point on your machine or it fails loudly.

This repository is the code layer. The full interactive certification (prose teaching, narrated videos, quizzes, in-browser labs, and the free certificate) lives on the site at [krypteiasec.com/academy](https://krypteiasec.com/academy). This is everything you clone, read, and run yourself in a terminal and in Jupyter.

## What is inside

```
labs/
  academy_llm.py        a tiny deterministic offline stand-in "LLM" the teaching labs import
  _models/              two REAL trained checkpoints + the training code (see below)
  foundations/          Course 0 labs
  <course>/             one folder of labs per course (prompt-engineering, rag, ...)
  lab-*.py              Course 1 (Build a Tiny LLM From Scratch) labs
notebooks/              151 Jupyter notebooks, one per lab, named by lab id
courses.export.json     the machine-readable catalog the site consumes (every course, chapter, lab)
COURSES.md              the human catalog: every course, chapter, and its lab + notebook
requirements.txt        torch, numpy, jupyter (only needed for the PyTorch + notebook labs)
```

The chapter videos and the interactive course are served on the site, not stored in this repo. See `COURSES.md` for the complete course-by-course, chapter-by-chapter map.

## How to use it

Two ways to work through a lab, both first-class.

### 1. Terminal

Every lab is a standalone script that prints an invariant proving the concept and exits 0.

```bash
python3 labs/prompt-engineering/pe2-few-shot.py
python3 labs/rag/rag6-rag-pipeline.py
python3 labs/ai-security/se8-red-team-harness.py
```

Most labs are pure standard library and need no install. The PyTorch labs (Course 1 LLM Fundamentals, Course 7 Training, Course 8 Transformers) and the trained models need `pip install -r requirements.txt`.

### 2. Jupyter notebooks

Each lab has a matching notebook, split into cells with the explanation above each step, so you run one idea at a time.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

Open, for example, `notebooks/rag6-rag-pipeline.ipynb` and run it top to bottom.

## The trained models (`labs/_models/`)

Two genuinely trained checkpoints you can load and generate from, not mocks:

- `tinygpt.pt`: a from-scratch character-level GPT (the same tiny architecture you build by hand in Course 1), trained until the loss actually falls and it produces coherent text.
- `lora_adapter.pt`: a real LoRA fine-tune of that base model on a downstream domain, with the base weights frozen.

```bash
python3 labs/_models/verify.py          # loads both, generates, asserts, prints MODELS OK
python3 labs/_models/train_tinygpt.py   # retrain the base from scratch (about 15s on Apple Silicon)
python3 labs/_models/finetune_lora.py   # retrain the LoRA adapter
```

## The offline "LLM" (`labs/academy_llm.py`)

The courses that teach prompt-in / text-out patterns (Prompt Engineering, Agents, Evals, App Engineering, Security, and more) import one small deterministic stand-in model. It is not a real model. It makes the mechanics visible and reproducible with no network and no API key: few-shot examples measurably steer its output, temperature changes sampling, a tool call routes, a judge scores, embeddings support retrieval. Every lesson that uses it can predict and assert on its behavior. When you are ready for a real model, the same interface points at a local or hosted LLM with the labs unchanged.

## The curriculum

The core track (Courses 0 to 13) takes you from Python and the math you need, through building a tiny LLM by hand, prompt engineering, RAG, agents and MCP, evaluation, application engineering, training and fine-tuning, the transformer internals, production and LLMOps, safety and security, multimodal, interview prep, and a capstone portfolio.

The applied track (Courses 14 to 18) adds building agents with Claude Code, the Claude API and Agent SDK, certification prep, setting up and running a personal AI operating system, and a flagship mastery course that teaches the full personalized setup the way its creator teaches it.

Full detail in `COURSES.md`.

## License

MIT. See `LICENSE`. The curriculum and code are free to read, run, fork, and teach from.
