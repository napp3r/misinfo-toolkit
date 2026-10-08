"""Streamlit demo for misinfo-toolkit.

Run locally with::

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

If ``models/model.joblib`` exists (e.g. trained on MiDe22 with ``misinfo train``)
it is used; otherwise a model is trained on the bundled synthetic sample data.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from misinfo_toolkit import __version__
from misinfo_toolkit.cues import find_cue_spans
from misinfo_toolkit.data import load_dataset, stratified_split
from misinfo_toolkit.evaluate import results_table, run_experiment
from misinfo_toolkit.models import MODELS, build_model, display_name, fake_probability, load_model

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "sample_tweets.csv"
SAVED_MODEL = ROOT / "models" / "model.joblib"
RESULTS_MD = ROOT / "results" / "mide22" / "results.md"

EXAMPLES = {
    "Conspiracy-style claim": "BREAKING: 5G towers secretly cause mass illness because they "
    "don't want you to know!! share before deleted #truth",
    "News-style report": "Officials report that the earthquake affected 3,400 people in the "
    "region, according to ministry data.",
}


@st.cache_resource
def get_model(name: str):
    if name == "saved" and SAVED_MODEL.exists():
        return load_model(SAVED_MODEL)
    df = load_dataset(SAMPLE_CSV)
    return build_model(name).fit(df["text"], df["label"])


@st.cache_data
def sample_benchmark() -> pd.DataFrame:
    return results_table(run_experiment(stratified_split(load_dataset(SAMPLE_CSV)), MODELS))


def highlight_cues(text: str) -> str:
    """HTML with causal cues wrapped in <mark>; everything else is escaped."""
    out, pos = [], 0
    for start, end, _ in find_cue_spans(text):
        out += [html.escape(text[pos:start]), f"<mark>{html.escape(text[start:end])}</mark>"]
        pos = end
    out.append(html.escape(text[pos:]))
    return "".join(out)


st.set_page_config(page_title="Misinformation detector", page_icon="🔎")
st.title("🔎 Tweet misinformation detector")
st.caption(f"misinfo-toolkit v{__version__} · TF-IDF baselines + causal cue features")

options = (["saved"] if SAVED_MODEL.exists() else []) + list(MODELS)
choice = st.sidebar.selectbox(
    "Model",
    options,
    format_func=lambda n: "Saved model (models/model.joblib)" if n == "saved" else display_name(n),
)
if choice != "saved":
    st.sidebar.info("Trained on the bundled **synthetic** demo data, not on real tweets.")

example = st.selectbox("Example", ["(type your own)", *EXAMPLES], key="example")
text = st.text_area(
    "Tweet text", value=EXAMPLES.get(example, ""), height=120, placeholder="Paste a tweet..."
)

if st.button("Analyse", type="primary", disabled=not text.strip()):
    score = float(fake_probability(get_model(choice), [text])[0])
    label = "Likely misinformation" if score >= 0.5 else "Likely truthful"
    col1, col2 = st.columns(2)
    col1.metric("Verdict", label)
    col2.metric("Misinformation score", f"{score:.2f}")
    st.progress(score)
    spans = find_cue_spans(text)
    st.subheader(f"Causal cues found: {len(spans)}")
    st.markdown(highlight_cues(text), unsafe_allow_html=True)

with st.expander("Benchmark results"):
    if RESULTS_MD.exists():
        st.markdown("**MiDe22 (English) test set**")
        st.markdown(RESULTS_MD.read_text())
    st.markdown("**Synthetic sample data (test split)**")
    st.dataframe(sample_benchmark(), hide_index=True)
