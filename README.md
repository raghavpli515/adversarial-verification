# Adversarial Verification System

A multi-agent system that verifies its own answers before returning them, instead of trusting a single LLM pass. Three agents — **Generator**, **Critic**, **Coordinator** — orchestrated with LangGraph, backed by retrieval over a curated corpus (Apollo program mission data), with an eval harness that measures whether the verification loop actually reduces unsupported claims and whether the system's stated confidence is honestly calibrated.

Built as a companion piece to a [multimodal trust-aware behavioral intelligence system](#) — the shared thread across both is **measurable reliability**, not just "the model works on my examples."

> **Status:** MVP complete. Eval numbers below are real, from three independent runs of the full 50-prompt set against `gpt-4o` — see [Results](#results) for the numbers and [Limitations](#limitations) for what they don't show.

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

The 50-prompt eval was run **three independent times** against `gpt-4o` (raw run output in `results/`). Repeated runs turned out to matter: an earlier pass on `gpt-4o-mini` showed pipeline accuracy swinging from 44% to 60% between otherwise-identical runs — noise large enough that a single run isn't a trustworthy number for this kind of eval. Numbers below are the mean across the three `gpt-4o` runs, with the per-run range shown so the reader can judge stability directly instead of trusting one point estimate.

| Metric | Baseline (Generator only) | Pipeline (Generator + Critic + Coordinator) |
|---|---|---|
| Accuracy | 0.787 (0.76 – 0.82) | 0.767 (0.74 – 0.78) |
| Unsupported-claim rate | 0.053 (0.04 – 0.06) | 0.040 (0.02 – 0.06) |
| False-escalation rate | n/a | 0.027 (0.00 – 0.054) |
| ECE, verbalized confidence | — | 0.199 (0.159 – 0.239) |
| ECE, engineered confidence | — | **0.064 (0.050 – 0.088)** |
| Mean latency / query | 2.85s | 4.77s (~1.7x) |
| Mean cost / query | $0.0034 | $0.0077 (~2.3x) |

**What this actually shows, stated plainly:**

- **Calibration is the real, reproducible result.** Engineered confidence holds an ECE of roughly 0.05–0.09 across all three independent runs; naive verbalized confidence (the Coordinator just self-reporting a number) is both worse *and* markedly less stable (0.16–0.24). A ~3x calibration improvement that survives three independent runs is a meaningfully stronger claim than a single favorable number would be.
- **The verification loop does not clearly improve raw accuracy.** Averaged across three runs, pipeline accuracy (76.7%) is marginally *below* the single-pass baseline (78.7%). This is reported without spin: the honest finding is that this architecture's value, at this scale, is in the confidence signal it produces, not in making the underlying answers more correct.
- **False escalation is low** (0–5.4% across runs) — the system rarely says "insufficient evidence" on a question the corpus could actually answer.
- **The critic loop costs a real, quantified overhead** (~2.3x cost, ~1.7x latency) for that calibration benefit — worth weighing against the alternative of shipping the naive verbalized-confidence number for free.

An earlier, more severe failure mode is also part of the honest story here, not hidden from it: three real bugs were found and fixed over the course of this eval (a critic prompt that produced self-contradictory findings on `gpt-4o-mini`; a chunking scheme where topically-broad chunks diluted individual facts in embedding space, observed directly as a chunk ranking 29th of 46 for a query it answered; and markdown header lines becoming their own near-duplicate chunks that flooded retrieval with noise). All three are described with reproduction details in the commit history and code comments — the eval harness is what surfaced each one.

## Limitations

- **The verification loop does not clearly improve raw accuracy at this scale.** Averaged across three runs, pipeline accuracy (76.7%) is marginally below single-pass baseline (78.7%) — see [Results](#results). The measured benefit of this architecture is calibration, not correctness, and that's a real constraint on how far the headline claim generalizes.
- **Small eval set.** 50 prompts split across four categories means roughly 12–13 items per category — enough to catch large, systematic failures (which is how the critic self-contradiction and retrieval-dilution bugs below were actually found), but too few for tight confidence intervals on any single metric. The three-run range reported alongside each number is a partial mitigation, not a substitute for a larger set.
- **The critic loop's reliability is sensitive to the underlying model, not just the prompt.** `gpt-4o-mini` showed real reasoning instability as the Critic — self-contradictory findings, and factually incorrect claims that evidence only "implies" a fact the evidence states outright with "because." Two rounds of targeted prompt engineering (an explicit definition of "unsupported" plus a worked example; removing "be adversarial, actively look for problems" framing) measurably reduced this but didn't eliminate it, and it's what motivated the move to `gpt-4o`. The architecture's trustworthiness claim is therefore conditional on the critic model being capable enough — a critic loop doesn't make an unreliable model reliable for free.
- **Retrieval quality was more fragile to chunking granularity than expected**, even on a ~20-document corpus. Two distinct bugs were found and fixed during this eval: topically-broad chunks diluting a specific fact's embedding similarity enough that the correct chunk ranked 29th of 46 for a query it directly answered, and bare markdown header lines becoming their own near-duplicate chunks that flooded top-k results with noise across every document. Both are fixed here, but the underlying lesson — dense single-vector embedding retrieval over hand-authored chunks is more sensitive to structural details than it looks — would need re-validating on a larger or messier corpus, not assumed away.
- **The engineered confidence formula is hand-tuned, not learned.** Its weights (retrieval-score component, critic-finding penalty, revision penalty) were chosen by judgment before looking at eval results, not fit against held-out data. That it out-calibrates verbalized confidence 3x is a real, reproduced finding; that it's the *best possible* formula is not claimed — a trained calibration model (e.g. logistic regression over the same signals) is the natural next step.
- **The LLM-judge is not fully independent of the system it grades.** The judge, the Critic, and the Generator all currently run on the same underlying model (`gpt-4o`), just with different prompts — a genuinely independent grader (a different model family) would rule out the possibility that a shared blind spot in one model makes both the system and the grader wrong in the same way.
- **One revision cycle** (`MAX_REVISION_CYCLES=1`) was never actually the bottleneck found in this eval — every observed failure traced back to the critic's judgment quality or retrieval quality, not to running out of revisions. Whether a higher cap would help on harder contradiction cases is untested, since none of this eval's failures were caused by exhausting the current cap.
- **Single-domain corpus.** Apollo program facts are unusually clean, well-structured, and free of genuine ambiguity — real-world corpora with conflicting sources, outdated information, or genuinely contested facts would test the Coordinator's "insufficient evidence" escalation logic far harder than this corpus can.

## Running locally

```bash
cp .env.example .env              # fill in OPENAI_API_KEY
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
