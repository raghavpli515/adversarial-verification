# Adversarial Verification System

A multi-agent system that verifies its own answers before returning them, instead of trusting a single LLM pass. Three agents — **Generator**, **Critic**, **Coordinator** — orchestrated with LangGraph, backed by retrieval over a curated corpus (Apollo program mission data), with an eval harness that measures whether the verification loop actually reduces unsupported claims and whether the system's stated confidence is honestly calibrated.

Built as a companion piece to a [multimodal trust-aware behavioral intelligence system](#) — the shared thread across both is **measurable reliability**, not just "the model works on my examples."

🎥 **[Watch a demo walkthrough](https://youtu.be/bDHBv1_jPOI)** — Generator → Critic → Coordinator in action, critic findings, and both confidence signals side by side.

> **Status:** MVP complete, including hybrid (dense + BM25) retrieval. Eval numbers below are real, from two batches of three independent 50-prompt runs each against `gpt-4o` — dense-only retrieval, then hybrid retrieval — see [Results](#results) for the full before/after comparison and [Limitations](#limitations) for what they don't show.

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
│  (Hybrid)    │                          └─────┬─────┘
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

**Retrieval is hybrid, not dense-only.** `HybridRetriever` fuses `ChromaRetriever` (dense, sentence-transformer embeddings) with `BM25Retriever` (sparse, exact term matching) via Reciprocal Rank Fusion — see `src/verification/retrieval/hybrid_retriever.py` for why RRF rather than a weighted score sum, and [Results](#results) for the accuracy/calibration tradeoff this produced measurably, not just in theory.

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

Two batches of three independent 50-prompt runs exist against `gpt-4o` (raw run output in `results/`): the original dense-only retrieval baseline, and a second batch after replacing dense-only retrieval with hybrid dense+BM25 retrieval (see Architecture). Repeated runs turned out to matter for both batches: an earlier pass on `gpt-4o-mini` showed pipeline accuracy swinging from 44% to 60% between otherwise-identical runs — noise large enough that a single run is never a trustworthy number for this kind of eval. Numbers below are the mean across each batch's three runs, with the per-run range shown so stability is visible rather than assumed.

| Metric | Dense-only retrieval (3-run) | Hybrid retrieval (3-run) |
|---|---|---|
| Baseline accuracy | 0.787 (0.76 – 0.82) | 0.880 (0.86 – 0.90) |
| Pipeline accuracy | 0.767 (0.74 – 0.78) | 0.867 (0.84 – 0.88) |
| Unsupported-claim rate (pipeline) | 0.040 (0.02 – 0.06) | 0.007 (0.00 – 0.02) |
| False-escalation rate | 0.027 (0.00 – 0.054) | 0.027 (identical in all 3 runs) |
| ECE, verbalized confidence | 0.199 (0.159 – 0.239) | 0.114 (0.088 – 0.149) |
| ECE, engineered confidence | **0.064 (0.050 – 0.088)** | 0.107 (0.071 – 0.127) |
| Mean latency / query (pipeline) | 4.77s | 5.87s |
| Mean cost / query (pipeline) | $0.0077 | $0.0076 |

**What this actually shows, stated plainly:**

- **Hybrid retrieval measurably improved accuracy and hallucination rate, consistently across all three runs.** Pipeline accuracy rose from 76.7% to 86.7%, baseline accuracy from 78.7% to 88.0%, and the unsupported-claim rate dropped from 4.0% to 0.7%. This tracks the documented mechanism: BM25 rescues chunks dense embeddings under-rank (see `corpus_loader.py`'s module docstring for the specific case), so the generator has better evidence to work with more often.
- **The calibration advantage — this project's original headline result — got weaker, and more importantly, unstable.** Engineered confidence's edge over naive verbalized confidence shrank from a clean 3x (0.064 vs 0.199, ranges never overlapping across three runs) to a near-tie (0.107 vs 0.114) that doesn't even hold direction consistently: engineered confidence lost to verbalized in two of the three hybrid-retrieval runs, then won clearly in the third. "Engineered confidence is reliably better calibrated" is no longer a claim these numbers support as cleanly as the dense-only batch did.
- **The root cause is diagnosed, not just observed.** Instrumenting the confidence formula directly (`get_engineered_confidence_components()`) showed that the critic-loop's finding rate collapsed to near-zero under hybrid retrieval (matching the unsupported-claim rate dropping to 0.7%), which pins `critic_component` at a near-constant 1.0 across most items — removing half of the formula's designed discriminative signal and leaving it dependent on `retrieval_component` alone, a single, noisier signal. See [Limitations](#limitations) for the full mechanism, including a distinct failure mode (unaddressed false premises in leading questions) the critic was never designed to catch.
- **Net assessment: a real tradeoff, not a strict improvement.** Hybrid retrieval is the better choice for this system's stated goal — reducing hallucination and improving raw correctness — at the cost of the calibration story that originally motivated building a separate engineered-confidence signal in the first place. Both halves of that tradeoff are reported here, not just the flattering one.
- **False-escalation rate stayed low and, unusually, landed on the exact same value in all three hybrid-retrieval runs** (2.7%) — the system still rarely says "insufficient evidence" on a question the corpus could actually answer.
- **Cost/latency overhead is comparable to the dense-only batch** (~1.5x latency, ~2.2x cost over baseline) — the extra local BM25 pass adds negligible cost; the LLM call overhead dominates either way.

An earlier, more severe failure mode is also part of the honest story here, not hidden from it: multiple real bugs were found and fixed across both phases of this eval — a critic prompt that produced self-contradictory findings on `gpt-4o-mini`; a chunking scheme where topically-broad chunks diluted individual facts in embedding space (a chunk ranking 29th of 46 for a query it directly answered); markdown header lines flooding retrieval with near-duplicate chunks; and, during hybrid retrieval development, a chunking scheme that stripped document-level context BM25 needs, plus a score-fusion bug that briefly regressed calibration to worse than the naive baseline (ECE 0.233) before being caught and fixed. All are described with reproduction details in the commit history, code comments, and [Limitations](#limitations) — the eval harness is what surfaced every one of them.

## Limitations

- **The verification loop does not clearly improve raw accuracy at this scale.** Averaged across three runs, pipeline accuracy (86.7% with hybrid retrieval, 76.7% with dense-only) is marginally below single-pass baseline in both cases (88.0% and 78.7% respectively) — see [Results](#results). The measured benefit of this architecture is in the confidence/hallucination signals it produces, not in making the underlying answers more correct, and that's a real constraint on how far the headline claim generalizes.
- **Small eval set.** 50 prompts split across four categories means roughly 12–13 items per category — enough to catch large, systematic failures (which is how the critic self-contradiction and retrieval-dilution bugs below were actually found), but too few for tight confidence intervals on any single metric. The three-run range reported alongside each number is a partial mitigation, not a substitute for a larger set.
- **The critic loop's reliability is sensitive to the underlying model, not just the prompt.** `gpt-4o-mini` showed real reasoning instability as the Critic — self-contradictory findings, and factually incorrect claims that evidence only "implies" a fact the evidence states outright with "because." Two rounds of targeted prompt engineering (an explicit definition of "unsupported" plus a worked example; removing "be adversarial, actively look for problems" framing) measurably reduced this but didn't eliminate it, and it's what motivated the move to `gpt-4o`. The architecture's trustworthiness claim is therefore conditional on the critic model being capable enough — a critic loop doesn't make an unreliable model reliable for free.
- **Retrieval quality was more fragile to chunking granularity than expected**, even on a ~20-document corpus. Two distinct bugs were found and fixed during this eval: topically-broad chunks diluting a specific fact's embedding similarity enough that the correct chunk ranked 29th of 46 for a query it directly answered, and bare markdown header lines becoming their own near-duplicate chunks that flooded top-k results with noise across every document. Both are fixed here, but the underlying lesson — dense single-vector embedding retrieval over hand-authored chunks is more sensitive to structural details than it looks — would need re-validating on a larger or messier corpus, not assumed away.
- **The engineered confidence formula is hand-tuned, not learned.** Its weights (retrieval-score component, critic-finding penalty, revision penalty) were chosen by judgment before looking at eval results, not fit against held-out data. That it out-calibrates verbalized confidence 3x is a real, reproduced finding; that it's the *best possible* formula is not claimed — a trained calibration model (e.g. logistic regression over the same signals) is the natural next step.
- **The critic verifies claims, not premises — a distinct failure mode found by instrumenting the confidence formula directly.** `get_engineered_confidence_components()` exposes the formula's three inputs individually; logging them across a 25-item diagnostic run showed `critic_component` pinned at exactly `1.0` for all 25 items, correct and wrong alike, confirmed by both the in-loop critic and an independently-called grading critic on the same wrong answers. The reason: the critic's job is checking whether each stated claim is evidence-supported, and on 3 of the 4 wrong answers in that run, every claim genuinely was — the actual defect was that the answer never addressed a false premise embedded in the query (e.g. "Given that Apollo 10 landed on the Moon, who was its Lunar Module pilot?" was answered correctly with "Eugene Cernan" but never noted Apollo 10 never landed). Since `critic_component` never varies, half of the formula's designed weight carries no discriminative signal in practice — a real, diagnosed mechanism, not a coding bug, and not something a `confidence.py` reweighting should paper over. A dedicated premise-verification step — checking the query's own assumptions against evidence, rather than just the answer's claims — is well-scoped future work, deliberately not built here: it's a new pipeline stage requiring its own prompt-engineering and validation cycle, not a patch to the existing critic, and out of scope for this project's timeline.
- **The LLM-judge is not fully independent of the system it grades.** The judge, the Critic, and the Generator all currently run on the same underlying model (`gpt-4o`), just with different prompts — a genuinely independent grader (a different model family) would rule out the possibility that a shared blind spot in one model makes both the system and the grader wrong in the same way.
- **One revision cycle** (`MAX_REVISION_CYCLES=1`) was never actually the bottleneck found in this eval — every observed failure traced back to the critic's judgment quality or retrieval quality, not to running out of revisions. Whether a higher cap would help on harder contradiction cases is untested, since none of this eval's failures were caused by exhausting the current cap.
- **Escalation triggers on empty retrieval, not on retrieval relevance.** `coordinator.py`'s `_decide` only escalates to "insufficient evidence" when zero chunks are retrieved — but every retriever here (dense, BM25, hybrid) always returns its top-k nearest chunks with no relevance cutoff, so that branch effectively never fires. For an out-of-corpus question ("What did the Apollo 18 crew find?"), retrieval still surfaces real chunks about cancelled missions, and the pipeline correctly answers "no evidence this happened, because the mission was cancelled" instead of escalating. Confirmed identical in eval runs both before and after hybrid retrieval: all 13 out-of-corpus prompts return `pipeline_status: "answered"` rather than the dataset's expected `"insufficient_evidence"` label, even though most are graded correct by the LLM-judge. This isn't miscalibration — false-escalation rate stays at 0% across runs — it's a mismatch between the eval dataset's binary status taxonomy and a pipeline that sometimes produces a better outcome (a grounded refusal) than a bare escalation would. A relevance-aware escalation rule (e.g. thresholding the same engineered confidence score already used for calibration, rather than an arbitrary new cutoff) is a natural next step, deliberately not built here without a larger eval set to validate it against.
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

Streamlit demo (visual walkthrough of the Generator → Critic → Coordinator flow, critic findings, and both confidence signals side by side — calls the pipeline directly, not through the API, so no separate `uvicorn` process is needed):

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

Intentionally local-only, not publicly deployed — every query is a real, billed OpenAI API call, and a public link would mean unmetered public access to that.

## Repo structure

```
src/verification/
  agents/{generator,critic,coordinator}.py   agent logic
  graph.py                                   LangGraph StateGraph wiring
  state.py                                   shared graph state schema
  confidence.py                              verbalized + engineered confidence, ECE
  retrieval/{base,vector_store,bm25_retriever,hybrid_retriever,corpus_loader}.py
src/api/                                     FastAPI app
eval/
  dataset/{adversarial_prompts.jsonl,corpus/}
  run_eval.py, judge.py, metrics.py
tests/                                       unit + integration tests
results/                                     eval run outputs
```
