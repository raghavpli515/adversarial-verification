"""Streamlit demo for the adversarial verification pipeline.

Calls `verification.graph.run_verification` directly — the same function
the API and eval harness use — rather than through the HTTP API, since it
gives access to state (critic findings, coordinator decision) that
`VerifyResponse` doesn't expose, with no separate `uvicorn` process needed.

Styling is a black + orange palette using color tokens pulled from
mistral.ai's published CSS (not their proprietary font or branding, just
the palette) — see `.streamlit/config.toml` for the base theme.

Run with: streamlit run demo/app.py
(after `python scripts/build_index.py` has built the Chroma index at least once)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from verification.graph import run_verification
from verification.retrieval.bm25_retriever import BM25Retriever
from verification.retrieval.hybrid_retriever import HybridRetriever
from verification.retrieval.vector_store import ChromaRetriever

# One example per eval category, all picked from items the pipeline answers
# correctly (see results/ for the full breakdown) — a demo is not the place
# to lead with a known failure mode; that story belongs in the README's
# Limitations section, which this app links to instead of re-litigating.
EXAMPLES = {
    "Answerable": "What was the name of the Apollo 11 Lunar Module?",
    "Leading question (false premise)": (
        "Since Buzz Aldrin was the first man on the Moon, what did he say when he stepped out?"
    ),
    "Out-of-corpus": "What did the Apollo 18 crew report finding at their landing site?",
    "Contradiction check": "Is it true that Apollo 11's Lunar Module was called 'Challenger'?",
}

st.set_page_config(page_title="Adversarial Verification Demo", layout="wide")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #101013; }

h1 {
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #F9F9FA;
}
h2, h3, h4 {
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #F9F9FA;
}
[data-testid="stCaptionContainer"] { color: #A1A1AA; }

/* Primary "Verify" button - solid orange pill */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #FA500F;
    color: #FFFFFF;
    border: none;
    border-radius: 999px;
    padding: 0.55rem 1.8rem;
    font-weight: 600;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background-color: #FF6529; color: #FFFFFF; }

/* Example / secondary buttons - dark ghost pill */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #1A1A1E;
    color: #F9F9FA;
    border: 1px solid #31313A;
    border-radius: 999px;
    font-weight: 500;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: #27272B;
    border-color: #474754;
    color: #F9F9FA;
}

/* Text input */
div[data-testid="stTextInput"] input {
    border-radius: 12px;
    border: 1px solid #31313A;
    background-color: #1A1A1E;
    color: #F9F9FA;
}

/* Card-style bordered containers (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1A1A1E;
    border: 1px solid #31313A !important;
    border-radius: 16px !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #1A1A1E;
    border: 1px solid #31313A;
    border-radius: 16px;
    padding: 1rem 1.2rem;
}
div[data-testid="stMetricLabel"] { color: #A1A1AA; }
div[data-testid="stMetricValue"] { color: #FA500F; font-weight: 700; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading retrieval index...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever(dense=ChromaRetriever(), sparse=BM25Retriever())


st.title("Adversarial Verification")
st.caption(
    "Generator → Critic → Coordinator pipeline over hybrid (dense + BM25) retrieval. "
    "See the project README for the full eval methodology, results, and honestly-reported limitations."
)
st.divider()

retriever = get_retriever()
if retriever.count() == 0:
    st.error("Corpus index is empty. Run `python scripts/build_index.py` first, then restart this app.")
    st.stop()

st.subheader("Try an example, or ask your own question")
example_cols = st.columns(len(EXAMPLES))
for col, (label, example_query) in zip(example_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["query"] = example_query

query = st.text_input("Question about the Apollo program corpus", key="query")
run_clicked = st.button("Verify", type="primary")

if run_clicked and query.strip():
    with st.spinner("Running Generator → Critic → Coordinator..."):
        state = run_verification(query, retriever)

    status = state["final_status"]
    with st.container(border=True):
        st.markdown("#### Result")
        if status == "answered":
            st.success("Status: answered")
        else:
            st.warning("Status: insufficient evidence")
        st.write(state["final_answer"])

        decision = state.get("coordinator_decision", "accept")
        revisions = state.get("revision_count", 0)
        if decision == "escalate":
            decision_label = f"Escalated to \"insufficient evidence\" after {revisions} revision(s)"
        elif revisions:
            decision_label = f"Accepted after {revisions} revision(s)"
        else:
            decision_label = "Accepted on first pass"
        st.caption(f"Coordinator decision: {decision_label}")

    with st.container(border=True):
        st.markdown("#### Confidence")
        conf_col1, conf_col2 = st.columns(2)
        conf_col1.metric("Verbalized (naive LLM self-report)", f"{state['confidence_verbalized']:.2f}")
        conf_col2.metric("Engineered (retrieval + critic signals)", f"{state['confidence_engineered']:.2f}")
        st.caption(
            "These two signals are compared for calibration accuracy (Expected Calibration "
            "Error) across a 50-prompt eval in the README's Results section — the point of "
            "this project is measuring which one is trustworthy, not just producing a number."
        )

    critic_report = state.get("critic_report", {"findings": []})
    findings = critic_report.get("findings", [])
    with st.expander(f"Critic findings ({len(findings)})", expanded=bool(findings)):
        if not findings:
            st.write("No issues found — every claim in the draft was checked against the retrieved evidence.")
        else:
            for f in findings:
                st.markdown(f"- **[{f['issue_type']}]** {f['claim']}  \n  _{f['explanation']}_")

    sources = state.get("retrieved_chunks", [])
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for c in sources:
            st.markdown(f"- `{c['chunk_id']}` ({c['source']}, score={c['score']:.3f}) — {c['text'][:200]}")

elif run_clicked:
    st.warning("Enter a question first.")
