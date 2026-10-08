import numpy as np
import pytest

from misinfo_toolkit.evaluate import BinaryMetrics, compute_metrics, results_table, run_experiment
from misinfo_toolkit.models import (
    MODELS,
    build_model,
    display_name,
    fake_probability,
    load_model,
    save_model,
)


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("bert")


@pytest.mark.parametrize("name", list(MODELS))
def test_models_fit_predict_and_score(name, sample_df):
    model = build_model(name)
    model.fit(sample_df["text"], sample_df["label"])
    texts = [
        "BREAKING: 5G towers secretly causes infertility, share before deleted!!",
        "Officials report that the earthquake affected 300 people, according to ministry data.",
    ]
    scores = fake_probability(model, texts)
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores[0] > 0.5 > scores[1]


def test_model_persistence(tmp_path, sample_df):
    model = build_model("logreg_cues").fit(sample_df["text"], sample_df["label"])
    path = tmp_path / "m" / "model.joblib"
    save_model(model, path)
    restored = load_model(path)
    np.testing.assert_allclose(
        fake_probability(restored, sample_df["text"][:5]),
        fake_probability(model, sample_df["text"][:5]),
    )


def test_compute_metrics_known_values():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m == BinaryMetrics(accuracy=0.5, precision_fake=0.5, recall_fake=0.5, f1_fake=0.5)


def test_compute_metrics_no_positive_predictions():
    m = compute_metrics([1, 0], [0, 0])
    assert m.precision_fake == 0.0 and m.recall_fake == 0.0


def test_run_experiment_beats_majority_baseline(sample_splits):
    results = run_experiment(sample_splits, MODELS)
    assert set(results) == set(MODELS)
    majority = max(sample_splits.test["label"].mean(), 1 - sample_splits.test["label"].mean())
    for name, r in results.items():
        assert r["model"] == display_name(name)
        assert r["test"]["accuracy"] > majority
        assert np.sum(r["confusion_matrix"]) == len(sample_splits.test)
    table = results_table(results)
    assert list(table["Model"]) == [display_name(n) for n in MODELS]
