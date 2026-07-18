# Course Catalog

> 19 courses, 151 chapters, 151 runnable labs, 151 notebooks, 151 videos.

## 0. Foundations: Python and Math for AI

From zero. Real Python you run yourself and only the math you need (vectors, matrices, probability, softmax, gradients). 8 chapters, live.

- **Ch 1: Your first Python: variables and types**  lab `f1-python-basics.py`, notebook `f1-python-basics.ipynb`, video `foundations-ch1.mp4`
- **Ch 2: Lists and loops: many values at once**  lab `f2-lists-loops.py`, notebook `f2-lists-loops.ipynb`, video `foundations-ch2.mp4`
- **Ch 3: Functions and dictionaries: reusable logic and lookups**  lab `f3-functions-dicts.py`, notebook `f3-functions-dicts.ipynb`, video `foundations-ch3.mp4`
- **Ch 4: Vectors: the math object at the center of AI**  lab `f4-vectors.py`, notebook `f4-vectors.ipynb`, video `foundations-ch4.mp4`
- **Ch 5: Matrices and the multiply that powers models**  lab `f5-matrices.py`, notebook `f5-matrices.ipynb`, video `foundations-ch5.mp4`
- **Ch 6: Probability and softmax: turning scores into choices**  lab `f6-probability-softmax.py`, notebook `f6-probability-softmax.ipynb`, video `foundations-ch6.mp4`
- **Ch 7: Derivatives and gradients: which way is downhill**  lab `f7-gradients.py`, notebook `f7-gradients.ipynb`, video `foundations-ch7.mp4`
- **Ch 8: Gradient descent: how models learn**  lab `f8-gradient-descent.py`, notebook `f8-gradient-descent.ipynb`, video `foundations-ch8.mp4`

## 1. LLM Fundamentals: Build a Tiny LLM From Scratch

Build a real language model by hand: tokenizer, embeddings, attention, transformer, training loop, generation. The depth vertical. 10 chapters, live.

- **Ch 1: Tokens: teaching a computer to read**  lab `lab-01-tokenizer.py`, notebook `lab-01-tokenizer.ipynb`, video `llm-fundamentals-ch1.mp4`
- **Ch 2: Embeddings: giving each token a vector of meaning**  lab `lab-02-embeddings.py`, notebook `lab-02-embeddings.ipynb`, video `llm-fundamentals-ch2.mp4`
- **Ch 3: The bigram model: your first real predictor**  lab `lab-03-bigram.py`, notebook `lab-03-bigram.ipynb`, video `llm-fundamentals-ch3.mp4`
- **Ch 4: Self attention: the idea that changed everything**  lab `lab-04-attention.py`, notebook `lab-04-attention.ipynb`, video `llm-fundamentals-ch4.mp4`
- **Ch 5: Multi head attention: many perspectives at once**  lab `lab-05-multihead.py`, notebook `lab-05-multihead.ipynb`, video `llm-fundamentals-ch5.mp4`
- **Ch 6: The transformer block: residuals, layer norm, and an MLP**  lab `lab-06-block.py`, notebook `lab-06-block.ipynb`, video `llm-fundamentals-ch6.mp4`
- **Ch 7: Assemble the GPT: stacking blocks into a model**  lab `lab-07-gpt.py`, notebook `lab-07-gpt.ipynb`, video `llm-fundamentals-ch7.mp4`
- **Ch 8: The training loop: how a model actually learns**  lab `lab-08-training.py`, notebook `lab-08-training.ipynb`, video `llm-fundamentals-ch8.mp4`
- **Ch 9: Generation: sampling text from your model**  lab `lab-09-generation.py`, notebook `lab-09-generation.ipynb`, video `llm-fundamentals-ch9.mp4`
- **Ch 10: Scaling, and what comes next**  lab `lab-10-scaling.py`, notebook `lab-10-scaling.ipynb`, video `llm-fundamentals-ch10.mp4`

## 2. Prompt Engineering

The first lever and highest-frequency daily skill: few-shot, chain-of-thought, structured output, caching, and eval-driven optimization.

- **Ch 1: Anatomy of a prompt: system, instructions, context**  lab `pe1-anatomy.py`, notebook `pe1-anatomy.ipynb`, video `prompt-engineering-ch1.mp4`
- **Ch 2: Few-shot prompting: teaching by example**  lab `pe2-few-shot.py`, notebook `pe2-few-shot.ipynb`, video `prompt-engineering-ch2.mp4`
- **Ch 3: Chain-of-thought and reasoning modes**  lab `pe3-chain-of-thought.py`, notebook `pe3-chain-of-thought.ipynb`, video `prompt-engineering-ch3.mp4`
- **Ch 4: Structured and JSON output**  lab `pe4-structured-output.py`, notebook `pe4-structured-output.ipynb`, video `prompt-engineering-ch4.mp4`
- **Ch 5: XML tagging and output shaping**  lab `pe5-xml-tagging.py`, notebook `pe5-xml-tagging.ipynb`, video `prompt-engineering-ch5.mp4`
- **Ch 6: Prompt caching and cost**  lab `pe6-prompt-caching.py`, notebook `pe6-prompt-caching.ipynb`, video `prompt-engineering-ch6.mp4`
- **Ch 7: Context engineering**  lab `pe7-context-engineering.py`, notebook `pe7-context-engineering.ipynb`, video `prompt-engineering-ch7.mp4`
- **Ch 8: Eval-driven prompt optimization**  lab `pe8-eval-optimization.py`, notebook `pe8-eval-optimization.ipynb`, video `prompt-engineering-ch8.mp4`

## 3. RAG and Embeddings

The #1 production pattern (70% of teams). Embeddings, chunking, vector search, re-ranking, end-to-end RAG, and the failure modes that bite.

- **Ch 1: Why models need external memory**  lab `rag1-external-memory.py`, notebook `rag1-external-memory.ipynb`, video `rag-ch1.mp4`
- **Ch 2: Embeddings for retrieval**  lab `rag2-embeddings.py`, notebook `rag2-embeddings.ipynb`, video `rag-ch2.mp4`
- **Ch 3: Vector search: nearest neighbors**  lab `rag3-vector-search.py`, notebook `rag3-vector-search.ipynb`, video `rag-ch3.mp4`
- **Ch 4: Chunking documents well**  lab `rag4-chunking.py`, notebook `rag4-chunking.ipynb`, video `rag-ch4.mp4`
- **Ch 5: Building the retrieval pipeline**  lab `rag5-pipeline.py`, notebook `rag5-pipeline.ipynb`, video `rag-ch5.mp4`
- **Ch 6: Retrieval augmented generation, end to end**  lab `rag6-rag-pipeline.py`, notebook `rag6-rag-pipeline.ipynb`, video `rag-ch6.mp4`
- **Ch 7: Failure modes: hallucination, stale context, bad chunks**  lab `rag7-failure-modes.py`, notebook `rag7-failure-modes.ipynb`, video `rag-ch7.mp4`
- **Ch 8: Evaluating a RAG system**  lab `rag8-evaluation.py`, notebook `rag8-evaluation.ipynb`, video `rag-ch8.mp4`

## 4. Agents, Tools and MCP

The 2026 frontier: the ReAct loop, tool and function calling, memory, and the Model Context Protocol (build an MCP server).

- **Ch 1: What an agent is: the model-in-a-loop**  lab `am1-what-is-an-agent.py`, notebook `am1-what-is-an-agent.ipynb`, video `agents-mcp-ch1.mp4`
- **Ch 2: The ReAct loop: reason, act, observe**  lab `am2-react-loop.py`, notebook `am2-react-loop.ipynb`, video `agents-mcp-ch2.mp4`
- **Ch 3: Designing tools the model can use**  lab `am3-tool-schemas.py`, notebook `am3-tool-schemas.ipynb`, video `agents-mcp-ch3.mp4`
- **Ch 4: Tool calling and function dispatch**  lab `am4-tool-dispatch.py`, notebook `am4-tool-dispatch.ipynb`, video `agents-mcp-ch4.mp4`
- **Ch 5: Memory and multi-step workflows**  lab `am5-memory.py`, notebook `am5-memory.ipynb`, video `agents-mcp-ch5.mp4`
- **Ch 6: The Model Context Protocol (MCP)**  lab `am6-mcp-protocol.py`, notebook `am6-mcp-protocol.ipynb`, video `agents-mcp-ch6.mp4`
- **Ch 7: Building an MCP server**  lab `am7-mcp-server.py`, notebook `am7-mcp-server.ipynb`, video `agents-mcp-ch7.mp4`
- **Ch 8: Guardrails on agents: least privilege and safety**  lab `am8-guarded-agent.py`, notebook `am8-guarded-agent.ipynb`, video `agents-mcp-ch8.mp4`

## 5. Evaluation and Testing

Evals replace unit tests. Golden datasets, code graders vs LLM-as-judge, regression gates, hallucination scoring. Table-stakes.

- **Ch 1: Why evals replace unit tests**  lab `ev1-why-evals.py`, notebook `ev1-why-evals.ipynb`, video `evals-ch1.mp4`
- **Ch 2: Building a golden dataset**  lab `ev2-golden-dataset.py`, notebook `ev2-golden-dataset.ipynb`, video `evals-ch2.mp4`
- **Ch 3: Code-based graders**  lab `ev3-code-graders.py`, notebook `ev3-code-graders.ipynb`, video `evals-ch3.mp4`
- **Ch 4: LLM-as-judge and its calibration**  lab `ev4-llm-as-judge.py`, notebook `ev4-llm-as-judge.ipynb`, video `evals-ch4.mp4`
- **Ch 5: Groundedness and hallucination scoring**  lab `ev5-groundedness.py`, notebook `ev5-groundedness.ipynb`, video `evals-ch5.mp4`
- **Ch 6: pass@k and statistical literacy**  lab `ev6-pass-at-k.py`, notebook `ev6-pass-at-k.ipynb`, video `evals-ch6.mp4`
- **Ch 7: Regression gates in CI**  lab `ev7-regression-gate.py`, notebook `ev7-regression-gate.ipynb`, video `evals-ch7.mp4`
- **Ch 8: Tracing and observability**  lab `ev8-tracing.py`, notebook `ev8-tracing.ipynb`, video `evals-ch8.mp4`

## 6. LLM Application Engineering

The literal daily job: APIs and SDKs, streaming, structured output, retries and rate limits, cost control, multi-provider routing, observability.

- **Ch 1: Calling model APIs and SDKs**  lab `ae1-api-client.py`, notebook `ae1-api-client.ipynb`, video `app-engineering-ch1.mp4`
- **Ch 2: Streaming responses (SSE) and TTFT**  lab `ae2-streaming.py`, notebook `ae2-streaming.ipynb`, video `app-engineering-ch2.mp4`
- **Ch 3: Structured output in production**  lab `ae3-structured-output.py`, notebook `ae3-structured-output.ipynb`, video `app-engineering-ch3.mp4`
- **Ch 4: Resilience: retries, backoff, rate limits**  lab `ae4-retries-backoff.py`, notebook `ae4-retries-backoff.ipynb`, video `app-engineering-ch4.mp4`
- **Ch 5: Cost control: routing, caching, budgets**  lab `ae5-cost-budget.py`, notebook `ae5-cost-budget.ipynb`, video `app-engineering-ch5.mp4`
- **Ch 6: Multi-provider architecture**  lab `ae6-multi-provider-router.py`, notebook `ae6-multi-provider-router.ipynb`, video `app-engineering-ch6.mp4`
- **Ch 7: Observability for LLM apps**  lab `ae7-observability.py`, notebook `ae7-observability.ipynb`, video `app-engineering-ch7.mp4`
- **Ch 8: Putting it together: a robust LLM app**  lab `ae8-robust-app.py`, notebook `ae8-robust-app.ipynb`, video `app-engineering-ch8.mp4`

## 7. Training and Fine-tuning

Shape a base model: datasets, loss curves, SFT and LoRA/QLoRA, DPO, and the judgment of when to fine-tune vs prompt vs RAG.

- **Ch 1: Datasets: the raw material a model learns from**  lab `tr1-datasets.py`, notebook `tr1-datasets.ipynb`, video `training-ch1.mp4`
- **Ch 2: Reading loss curves: overfitting and validation**  lab `tr2-loss-curves.py`, notebook `tr2-loss-curves.ipynb`, video `training-ch2.mp4`
- **Ch 3: Supervised fine-tuning (SFT)**  lab `tr3-sft.py`, notebook `tr3-sft.ipynb`, video `training-ch3.mp4`
- **Ch 4: LoRA from scratch: freeze the base, learn a little**  lab `tr4-lora-from-scratch.py`, notebook `tr4-lora-from-scratch.ipynb`, video `training-ch4.mp4`
- **Ch 5: LoRA and QLoRA in practice: base vs adapter**  lab `tr5-lora-base-vs-adapter.py`, notebook `tr5-lora-base-vs-adapter.ipynb`, video `training-ch5.mp4`
- **Ch 6: Preference tuning with DPO**  lab `tr6-dpo-preference.py`, notebook `tr6-dpo-preference.ipynb`, video `training-ch6.mp4`
- **Ch 7: Evaluating a fine-tune without fooling yourself**  lab `tr7-evaluate-finetune.py`, notebook `tr7-evaluate-finetune.ipynb`, video `training-ch7.mp4`
- **Ch 8: Putting it together: fine-tune vs prompt vs RAG**  lab `tr8-finetune-vs-prompt-vs-rag.py`, notebook `tr8-finetune-vs-prompt-vs-rag.ipynb`, video `training-ch8.mp4`

## 8. Transformers Deep Dive

Explain the internals cold: positional encodings (RoPE), attention variants, BPE, KV cache, scaling laws. The T-shape depth vertical.

- **Ch 1: Positional encodings: from sinusoidal to RoPE**  lab `tf1-positional-encoding.py`, notebook `tf1-positional-encoding.ipynb`, video `transformers-ch1.mp4`
- **Ch 2: Attention variants: MHA, MQA, and GQA**  lab `tf2-attention-variants.py`, notebook `tf2-attention-variants.ipynb`, video `transformers-ch2.mp4`
- **Ch 3: Tokenization at scale: byte-pair encoding**  lab `tf3-bpe.py`, notebook `tf3-bpe.ipynb`, video `transformers-ch3.mp4`
- **Ch 4: The KV cache and efficient inference**  lab `tf4-kv-cache.py`, notebook `tf4-kv-cache.ipynb`, video `transformers-ch4.mp4`
- **Ch 5: Normalization and residuals in depth**  lab `tf5-layernorm-residual.py`, notebook `tf5-layernorm-residual.ipynb`, video `transformers-ch5.mp4`
- **Ch 6: The full block, and a hard proof of causality**  lab `tf6-block-causality.py`, notebook `tf6-block-causality.ipynb`, video `transformers-ch6.mp4`
- **Ch 7: Scaling laws**  lab `tf7-scaling-laws.py`, notebook `tf7-scaling-laws.ipynb`, video `transformers-ch7.mp4`
- **Ch 8: Why transformers work**  lab `tf8-why-transformers-work.py`, notebook `tf8-why-transformers-work.ipynb`, video `transformers-ch8.mp4`

## 9. AI Engineering in Production

Ship and operate: serving, quantization (AWQ/GGUF), vLLM/KV cache, eval-gated CI/CD, cost and drift monitoring, LLMOps.

- **Ch 1: Serving models: latency and throughput**  lab `pr1-serving-batching.py`, notebook `pr1-serving-batching.ipynb`, video `production-ch1.mp4`
- **Ch 2: Quantization: running models cheaply**  lab `pr2-quantization.py`, notebook `pr2-quantization.ipynb`, video `production-ch2.mp4`
- **Ch 3: The KV cache and the memory cost of serving**  lab `pr3-kv-cache-cost.py`, notebook `pr3-kv-cache-cost.ipynb`, video `production-ch3.mp4`
- **Ch 4: Evals as a production gate**  lab `pr4-eval-gate.py`, notebook `pr4-eval-gate.ipynb`, video `production-ch4.mp4`
- **Ch 5: Cost monitoring and budgets**  lab `pr5-cost-monitor.py`, notebook `pr5-cost-monitor.ipynb`, video `production-ch5.mp4`
- **Ch 6: Drift and quality monitoring**  lab `pr6-drift-detector.py`, notebook `pr6-drift-detector.ipynb`, video `production-ch6.mp4`
- **Ch 7: The LLMOps lifecycle: versioning and rollback**  lab `pr7-llmops-lifecycle.py`, notebook `pr7-llmops-lifecycle.ipynb`, video `production-ch7.mp4`
- **Ch 8: Shipping: serving, gating, and monitoring as one system**  lab `pr8-capstone.py`, notebook `pr8-capstone.ipynb`, video `production-ch8.mp4`

## 10. AI Safety and Security

The OWASP LLM Top 10, prompt-injection offense and defense, guardrails, and red-teaming your own app. Ada's differentiator.

- **Ch 1: The LLM attack surface and the OWASP LLM Top 10**  lab `se1-attack-surface.py`, notebook `se1-attack-surface.ipynb`, video `ai-security-ch1.mp4`
- **Ch 2: Prompt injection: direct (LLM01)**  lab `se2-direct-injection.py`, notebook `se2-direct-injection.ipynb`, video `ai-security-ch2.mp4`
- **Ch 3: Indirect injection and RAG poisoning (LLM08)**  lab `se3-indirect-injection.py`, notebook `se3-indirect-injection.ipynb`, video `ai-security-ch3.mp4`
- **Ch 4: System-prompt extraction and leakage (LLM07)**  lab `se4-system-prompt-extraction.py`, notebook `se4-system-prompt-extraction.ipynb`, video `ai-security-ch4.mp4`
- **Ch 5: Guardrails and input/output filtering**  lab `se5-guardrails.py`, notebook `se5-guardrails.ipynb`, video `ai-security-ch5.mp4`
- **Ch 6: Tool and agent abuse prevention (LLM06)**  lab `se6-tool-abuse.py`, notebook `se6-tool-abuse.ipynb`, video `ai-security-ch6.mp4`
- **Ch 7: Sensitive-data disclosure and exfiltration (LLM02)**  lab `se7-data-exfiltration.py`, notebook `se7-data-exfiltration.ipynb`, video `ai-security-ch7.mp4`
- **Ch 8: Red-team your own app: the attack harness**  lab `se8-red-team-harness.py`, notebook `se8-red-team-harness.ipynb`, video `ai-security-ch8.mp4`

## 11. Multimodal AI

Vision-language, image generation, speech (STT/TTS), and multimodal RAG. The converging edge.

- **Ch 1: Image representation: pixels to patches**  lab `mm1-image-patches.py`, notebook `mm1-image-patches.ipynb`, video `multimodal-ch1.mp4`
- **Ch 2: Vision-language models: one shared space**  lab `mm2-clip-matching.py`, notebook `mm2-clip-matching.ipynb`, video `multimodal-ch2.mp4`
- **Ch 3: Image generation: denoising step by step**  lab `mm3-image-generation.py`, notebook `mm3-image-generation.ipynb`, video `multimodal-ch3.mp4`
- **Ch 4: Speech: STT and TTS concepts**  lab `mm4-speech.py`, notebook `mm4-speech.ipynb`, video `multimodal-ch4.mp4`
- **Ch 5: Multimodal RAG: retrieving over mixed media**  lab `mm5-multimodal-rag.py`, notebook `mm5-multimodal-rag.ipynb`, video `multimodal-ch5.mp4`
- **Ch 6: Putting multimodal together: a tiny assistant**  lab `mm6-assistant.py`, notebook `mm6-assistant.ipynb`, video `multimodal-ch6.mp4`

## 12. Interview Prep and System Design

Ace the interview: the real loop, from-scratch coding, LLM system design, take-home patterns, behavioral, and question banks.

- **Ch 1: The real interview loop**  lab `ip1-cosine-retrieval.py`, notebook `ip1-cosine-retrieval.ipynb`, video `interview-prep-ch1.mp4`
- **Ch 2: Practical coding rounds**  lab `ip2-tokenizer.py`, notebook `ip2-tokenizer.ipynb`, video `interview-prep-ch2.mp4`
- **Ch 3: From-scratch ML coding**  lab `ip3-attention.py`, notebook `ip3-attention.ipynb`, video `interview-prep-ch3.mp4`
- **Ch 4: ML and LLM theory with judgment**  lab `ip4-rag-vs-finetune-decision.py`, notebook `ip4-rag-vs-finetune-decision.ipynb`, video `interview-prep-ch4.mp4`
- **Ch 5: LLM system design**  lab `ip5-sizing-calculator.py`, notebook `ip5-sizing-calculator.ipynb`, video `interview-prep-ch5.mp4`
- **Ch 6: Take-home projects that pass**  lab `ip6-take-home-grader.py`, notebook `ip6-take-home-grader.ipynb`, video `interview-prep-ch6.mp4`
- **Ch 7: Behavioral and safety mindset**  lab `ip7-sampling.py`, notebook `ip7-sampling.ipynb`, video `interview-prep-ch7.mp4`
- **Ch 8: Question banks and mock rounds**  lab `ip8-softmax.py`, notebook `ip8-softmax.ipynb`, video `interview-prep-ch8.mp4`

## 13. Capstone Projects

What gets you hired: 3-5 deployed, evaluated builds. RAG assistant, tool-calling agent, eval pipeline, MCP assistant, self-red-teamed app.

- **Ch 1: The portfolio that gets hired**  lab `cap1-portfolio-rubric.py`, notebook `cap1-portfolio-rubric.ipynb`, video `capstones-ch1.mp4`
- **Ch 2: Capstone 1: a deployed RAG assistant that cites its sources**  lab `cap2-rag-assistant.py`, notebook `cap2-rag-assistant.ipynb`, video `capstones-ch2.mp4`
- **Ch 3: Capstone 2: a tool-calling agent that solves a multi-step task**  lab `cap3-tool-agent.py`, notebook `cap3-tool-agent.ipynb`, video `capstones-ch3.mp4`
- **Ch 4: Capstone 3: a reusable eval pipeline that gates a build**  lab `cap4-eval-pipeline.py`, notebook `cap4-eval-pipeline.ipynb`, video `capstones-ch4.mp4`
- **Ch 5: Capstone 4: an MCP-connected assistant**  lab `cap5-mcp-assistant.py`, notebook `cap5-mcp-assistant.ipynb`, video `capstones-ch5.mp4`
- **Ch 6: Capstone 5: a self-red-teamed, secured LLM app**  lab `cap6-red-team-app.py`, notebook `cap6-red-team-app.ipynb`, video `capstones-ch6.mp4`
- **Ch 7: Packaging: manifest, READMEs, demo links, and the write-up**  lab `cap7-packaging.py`, notebook `cap7-packaging.ipynb`, video `capstones-ch7.mp4`

## 14. Claude Code and Agentic Builds

Build agents the way Claude Code does: the agentic loop, the tool contract, ReAct, subagents and delegation, MCP, hooks and guardrails, orchestrating a fleet, and a complete agentic workflow. 8 chapters, offline labs.

- **Ch 1: What Claude Code is, and the agentic loop**  lab `cc1-agentic-loop.py`, notebook `cc1-agentic-loop.ipynb`, video `claude-code-agents-ch1.mp4`
- **Ch 2: Defining tools: the function-calling contract**  lab `cc2-tools.py`, notebook `cc2-tools.ipynb`, video `claude-code-agents-ch2.mp4`
- **Ch 3: The ReAct pattern in practice**  lab `cc3-react.py`, notebook `cc3-react.ipynb`, video `claude-code-agents-ch3.mp4`
- **Ch 4: Subagents and delegation**  lab `cc4-subagents.py`, notebook `cc4-subagents.ipynb`, video `claude-code-agents-ch4.mp4`
- **Ch 5: MCP integration: why and how**  lab `cc5-mcp.py`, notebook `cc5-mcp.ipynb`, video `claude-code-agents-ch5.mp4`
- **Ch 6: Hooks and guardrails**  lab `cc6-hooks.py`, notebook `cc6-hooks.ipynb`, video `claude-code-agents-ch6.mp4`
- **Ch 7: Orchestrating a fleet**  lab `cc7-fleet.py`, notebook `cc7-fleet.ipynb`, video `claude-code-agents-ch7.mp4`
- **Ch 8: Building a complete agentic workflow**  lab `cc8-workflow.py`, notebook `cc8-workflow.ipynb`, video `claude-code-agents-ch8.mp4`

## 15. Claude Code SDK and API

Drive Claude from code: the Messages API and Agent SDK end to end. Typed responses, system and roles, streaming, tool use, the agent loop (tool_runner), structured output, prompt caching and token counting, then a small shippable app.

- **Ch 1: The Messages API: one endpoint, one shape**  lab `sdk1-messages-api.py`, notebook `sdk1-messages-api.ipynb`, video `claude-code-sdk-ch1.mp4`
- **Ch 2: System prompts, roles, and a stateless API**  lab `sdk2-system-and-roles.py`, notebook `sdk2-system-and-roles.ipynb`, video `claude-code-sdk-ch2.mp4`
- **Ch 3: Streaming responses**  lab `sdk3-streaming.py`, notebook `sdk3-streaming.ipynb`, video `claude-code-sdk-ch3.mp4`
- **Ch 4: Tool use and function calling**  lab `sdk4-tool-use.py`, notebook `sdk4-tool-use.ipynb`, video `claude-code-sdk-ch4.mp4`
- **Ch 5: The agent loop and the tool runner**  lab `sdk5-agent-loop.py`, notebook `sdk5-agent-loop.ipynb`, video `claude-code-sdk-ch5.mp4`
- **Ch 6: Structured output you can trust**  lab `sdk6-structured-output.py`, notebook `sdk6-structured-output.ipynb`, video `claude-code-sdk-ch6.mp4`
- **Ch 7: Prompt caching and token counting**  lab `sdk7-caching-tokens.py`, notebook `sdk7-caching-tokens.ipynb`, video `claude-code-sdk-ch7.mp4`
- **Ch 8: Build a small app with the SDK**  lab `sdk8-build-app.py`, notebook `sdk8-build-app.ipynb`, video `claude-code-sdk-ch8.mp4`

## 16. CCA-F Certification Prep

Pass the Claude certification: the weighted exam domains, scenarios, and recurring wrong-answer traps, each drilled with a runnable lab and a mock exam scored at the real pass line. 8 chapters.

- **Ch 1: The exam blueprint: format, domains, and how to pass**  lab `cf1-exam-blueprint.py`, notebook `cf1-exam-blueprint.ipynb`, video `cca-f-ch1.mp4`
- **Ch 2: Domain 1: the agent loop and stop_reason**  lab `cf2-agent-loop.py`, notebook `cf2-agent-loop.ipynb`, video `cca-f-ch2.mp4`
- **Ch 3: Domain 1: orchestration, hooks, and escalation**  lab `cf3-orchestration.py`, notebook `cf3-orchestration.ipynb`, video `cca-f-ch3.mp4`
- **Ch 4: Domain 3: Claude Code configuration and workflows**  lab `cf4-claude-code-config.py`, notebook `cf4-claude-code-config.ipynb`, video `cca-f-ch4.mp4`
- **Ch 5: Domain 2: prompt engineering and structured output**  lab `cf5-structured-output.py`, notebook `cf5-structured-output.ipynb`, video `cca-f-ch5.mp4`
- **Ch 6: Domain 4: tool design and MCP integration**  lab `cf6-tool-design.py`, notebook `cf6-tool-design.ipynb`, video `cca-f-ch6.mp4`
- **Ch 7: Domain 5: context management and reliability**  lab `cf7-context-reliability.py`, notebook `cf7-context-reliability.ipynb`, video `cca-f-ch7.mp4`
- **Ch 8: The eight scenarios, the traps, and exam day**  lab `cf8-scenario-drill.py`, notebook `cf8-scenario-drill.ipynb`, video `cca-f-ch8.mp4`

## 17. LifeOS: Setup to Advanced

The personal AI operating system end to end: current state to ideal state, install, TELOS, the Algorithm, skills, hooks, Pulse, and composing it into a daily OS. 8 chapters.

- **Ch 1: What LifeOS is and why it exists**  lab `lo1-current-to-ideal.py`, notebook `lo1-current-to-ideal.ipynb`, video `lifeos-ch1.mp4`
- **Ch 2: Install and setup: from zero to a running system**  lab `lo2-install-preflight.py`, notebook `lo2-install-preflight.ipynb`, video `lifeos-ch2.mp4`
- **Ch 3: TELOS: naming your current state and your ideal state**  lab `lo3-telos-parse.py`, notebook `lo3-telos-parse.ipynb`, video `lifeos-ch3.mp4`
- **Ch 4: The Algorithm: how work actually gets done**  lab `lo4-algorithm-phases.py`, notebook `lo4-algorithm-phases.ipynb`, video `lifeos-ch4.mp4`
- **Ch 5: Skills: deterministic units of capability**  lab `lo5-skill-frontmatter.py`, notebook `lo5-skill-frontmatter.ipynb`, video `lifeos-ch5.mp4`
- **Ch 6: Hooks and automation: acting without being asked**  lab `lo6-hook-dispatch.py`, notebook `lo6-hook-dispatch.ipynb`, video `lifeos-ch6.mp4`
- **Ch 7: The Pulse dashboard: the face of the system**  lab `lo7-pulse-health.py`, notebook `lo7-pulse-health.ipynb`, video `lifeos-ch7.mp4`
- **Ch 8: Advanced: composing a daily operating system**  lab `lo8-daily-os.py`, notebook `lo8-daily-os.ipynb`, video `lifeos-ch8.mp4`

## 18. LifeOS Mastery: The Complete Setup

The flagship LifeOS course, taught the way its creator teaches it: why before how, and the full personalized setup end to end. Install and upgrade, name your DA, identity and a quantified 12-trait personality, TELOS in two parts, the /interview, migrate your context, and run it all through the Algorithm as a daily OS. 8 chapters, offline labs.

- **Ch 1: What LifeOS is, and the whole setup in one pass**  lab `lm1-install-preflight.py`, notebook `lm1-install-preflight.ipynb`, video `lifeos-mastery-ch1.mp4`
- **Ch 2: Name your DA**  lab `lm2-name-your-da.py`, notebook `lm2-name-your-da.ipynb`, video `lifeos-mastery-ch2.mp4`
- **Ch 3: Give it identity and personality**  lab `lm3-personality-builder.py`, notebook `lm3-personality-builder.ipynb`, video `lifeos-mastery-ch3.mp4`
- **Ch 4: TELOS, part one: the foundational why**  lab `lm4-goal-validator.py`, notebook `lm4-goal-validator.ipynb`, video `lifeos-mastery-ch4.mp4`
- **Ch 5: TELOS, part two: narratives, beliefs, and current to ideal**  lab `lm5-current-to-ideal.py`, notebook `lm5-current-to-ideal.ipynb`, video `lifeos-mastery-ch5.mp4`
- **Ch 6: The /interview flow**  lab `lm6-interview-state-machine.py`, notebook `lm6-interview-state-machine.ipynb`, video `lifeos-mastery-ch6.mp4`
- **Ch 7: Personalize the rest of you**  lab `lm7-migrate-classify.py`, notebook `lm7-migrate-classify.ipynb`, video `lifeos-mastery-ch7.mp4`
- **Ch 8: Advanced use: the daily operating system**  lab `lm8-maturity-and-algorithm.py`, notebook `lm8-maturity-and-algorithm.ipynb`, video `lifeos-mastery-ch8.mp4`

