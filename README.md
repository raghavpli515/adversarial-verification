# Adversarial Verification System

A multi-agent system that verifies its own answers before returning them, instead of trusting a single LLM pass. Three agents — **Generator**, **Critic**, **Coordinator** — orchestrated with LangGraph, backed by retrieval over a curated corpus (Apollo program mission data), with an eval harness that measures whether the verification loop actually reduces unsupported claims and whether the system's stated confidence is honestly calibrated.

Built as a companion piece to a [multimodal trust-aware behavioral intelligence system](#) — the shared thread across both is **measurable reliability**, not just "the model works on my examples."

> **Status:** work in progress. This README will be updated with final eval numbers once the harness has run against the full 50-prompt adversarial set. Numbers below are placeholders until then — see [Results](#results).

---

## Problem framing

Single-pass LLM answers over retrieved context fail in three specific, measurable ways:
1. **Unsupported claims** — the model states something not present in the retrieved evidence (hallucination on top of RAG, not despite it).
2. **Miscalibrated confidence** — the model sounds equally confident whether it's right or fabricating.
3. **Overreach on missing evidence** — instead of saying "I don't know," the model answers anyway from parametric knowledge, silently mixing corpus facts with guesses.

This project builds a verification loop that adversarially checks a draft answer against its own evidence before returning it, and — this is the part that's usually skipped — **measures whether the loop actually helps**, with a held-out adversarial prompt set designed specifically to induce these three failure modes.

## Architecture

```
 query
   │
   ▼
┌─────────────┐     retrieved chunks     ┌───────────┐
│  Retriever   │ ────────────────────────▶│ Generator │
│  (Chroma)    │                          └─────┬─────┘
└─────────────┘                                 │ draft answer + citations
                                                 ▼
                                           ┌───────────┐
                                           │  Critic   │  checks draft against
                                           │           │  the SAME retrieved chunks
                                           └─────┬─────┘
                                                 │ findings (unsupported claims,
                                                 │ contradictions, overconfidence)
                                                 ▼
                                         ┌───────────────┐
                                         │  Coordinator  │
                                         └───────┬───────┘
                                    accept │ revise │ escalate
                                           │    │        │
                                           │    └──▶ back to Generator (max 1x)
                                           ▼             ▼
                                    final answer   "insufficient evidence"
                                    + confidence
```

The Generator and Critic never talk directly — everything routes through shared graph state, and the Coordinator is the only node that decides control flow (accept / send back for one revision / escalate to "insufficient evidence"). This is deliberately not "three agents chatting" — the Coordinator's decision is a hybrid of deterministic rules (finding counts, retrieval quality) and one LLM judgment call, not another unconstrained LLM conversation. See `src/verification/agents/coordinator.py` for the exact logic.

**Why LangGraph:** the critic → coordinator → (back to generator) cycle needs actual control flow with a hard iteration cap — not expressible as a linear chain. LangGraph models the pipeline as nodes over one shared, typed state object (`src/verification/state.py`) with conditional edges deciding what runs next, and a built-in recursion limit so a misbehaving loop can't run forever.

## Corpus

~20 curated documents on the Apollo program (mission overviews for Apollo 1–17, program history, key crew). Chosen deliberately narrow so adversarial prompts could be hand-authored against known gaps — including the fact that Apollo 18–20 were cancelled, which is a built-in, verifiable source of "this should be flagged as insufficient evidence" test cases.

## Eval methodology

50 hand-authored adversarial prompts across four categories, each with a gold label:

| Category | Expected behavior |
|---|---|
| Answerable, directly supported | Correct answer, high confidence |
| Leading / paraphrase-trap (answerable) | Correct answer, not misled by the framing |
| Out-of-corpus / unanswerable | "Insufficient evidence," not a guess |
| Contradiction-inducing | Flags the conflict rather than silently picking a side |

Every prompt is run through **both** a single-pass Generator-only baseline and the full Generator→Critic→Coordinator pipeline, so every metric below is reported as baseline vs. pipeline, not in isolation. Grading uses a separate LLM-judge call (same model, independent prompt, not one of the three pipeline agents) scoring the answer against the gold label — exact string match doesn't work for open-ended answers.

Metrics tracked per run (see `eval/metrics.py`), logged to MLflow per run for reproducibility:

- **Unsupported-claim rate** — baseline vs. pipeline
- **Calibration** — Expected Calibration Error (ECE) + reliability diagram, comparing naive verbalized confidence (the Coordinator just self-reporting a number) against an engineered confidence score built from retrieval similarity, critic finding counts, and revision history
- **False-escalation rate** — how often the system says "insufficient evidence" on a prompt that was actually answerable from the corpus
- **Latency / cost overhead** — pipeline vs. single-pass baseline, per query

## Results

*Populated after `eval/run_eval.py` completes against the full prompt set — see `results/` for raw run output once available.*

| Metric | Baseline (Generator only) | Pipeline (Generator + Critic + Coordinator) |
|---|---|---|
| Unsupported-claim rate | — | — |
| ECE (verbalized confidence) | — | — |
| ECE (engineered confidence) | — | — |
| False-escalation rate | n/a | — |
| p50 latency | — | — |
| p50 cost / query | — | — |

## Limitations

*(To be written honestly once the eval numbers exist — anticipated topics: single-domain corpus doesn't test generalization; 50 prompts is a small eval set for confidence-interval claims; the engineered confidence formula is hand-tuned, not learned; one revision cycle may not be enough for harder contradiction cases; LLM-judge grading has its own calibration issues that aren't independently validated here.)*

## Running locally

```bash
cp .env.example .env              # fill in ANTHROPIC_API_KEY
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/build_index.py     # builds the Chroma index from eval/dataset/corpus/
uvicorn api.main:app --reload     # FastAPI on :8000, POST /verify
```

> **Windows note:** if `pip install -e ".[dev]"` fails with `WinError 206: filename or extension too long`, it's Windows' 260-character `MAX_PATH` limit, triggered by `torch`'s own deeply nested license-file tree (a transitive dependency of `sentence-transformers`) combined with a long project folder path — not a broken install. Fix by either enabling [Windows long-path support](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation) (needs admin rights: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`, then reboot), or by creating `.venv` at a short path outside the project instead of inside it, e.g. `python -m venv C:\venvs\adversarial-verification` — an editable install (`pip install -e .`) still points back at the project source regardless of where the venv itself lives.

```bash
pytest                             # unit + integration tests
python eval/run_eval.py            # full eval harness, baseline vs pipeline
```

Docker:

```bash
docker build -t adversarial-verification .
docker run -p 8000:8000 --env-file .env adversarial-verification
```

## Repo structure

```
src/verification/
  agents/{generator,critic,coordinator}.py   agent logic
  graph.py                                   LangGraph StateGraph wiring
  state.py                                   shared graph state schema
  confidence.py                              verbalized + engineered confidence, ECE
  retrieval/{base,vector_store,corpus_loader}.py
src/api/                                     FastAPI app
eval/
  dataset/{adversarial_prompts.jsonl,corpus/}
  run_eval.py, judge.py, metrics.py
tests/                                       unit + integration tests
results/                                     eval run outputs
```
