"""Model factory: TF-IDF linear baselines and a TF-IDF + causal cues variant."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from . import RANDOM_SEED
from .cues import extract_cue_matrix
from .preprocess import tokenize


def make_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """TF-IDF over uni- and bi-grams produced by :func:`~misinfo_toolkit.preprocess.tokenize`."""
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,  # tokenize() already lower-cases
        ngram_range=(1, 2),
        max_features=max_features,
    )


def _cue_features(texts: Sequence[str]) -> np.ndarray:
    return extract_cue_matrix(list(texts))


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)


def _build_logreg(max_features: int, seed: int) -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer(max_features)), ("clf", _logreg(seed))])


def _build_svm(max_features: int, seed: int) -> Pipeline:
    return Pipeline(
        [("tfidf", make_vectorizer(max_features)), ("clf", LinearSVC(random_state=seed))]
    )


def _build_logreg_cues(max_features: int, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            ("tfidf", make_vectorizer(max_features)),
            ("cues", FunctionTransformer(_cue_features)),
        ]
    )
    return Pipeline([("features", features), ("clf", _logreg(seed))])


MODELS: dict[str, tuple[str, Callable[[int, int], Pipeline]]] = {
    "logreg": ("Logistic Regression", _build_logreg),
    "svm": ("Linear SVM", _build_svm),
    "logreg_cues": ("LogReg + Causal Cues", _build_logreg_cues),
}


def display_name(name: str) -> str:
    return MODELS[name][0]


def build_model(name: str, *, max_features: int = 5000, seed: int = RANDOM_SEED) -> Pipeline:
    """Create an unfitted scikit-learn pipeline by its short name (see :data:`MODELS`)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name][1](max_features, seed)


def fake_probability(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """Score in [0, 1] that each text is misinformation.

    Uses ``predict_proba`` when available; for margin classifiers (Linear SVM)
    the decision function is squashed with a logistic sigmoid, so the value is
    a confidence score rather than a calibrated probability.
    """
    texts = list(texts)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(texts)))


def save_model(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
