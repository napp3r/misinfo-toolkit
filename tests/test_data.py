import json

import pandas as pd
import pytest

from misinfo_toolkit.data import (
    Splits,
    binarize_mide22,
    load_dataset,
    load_mide22_tsv,
    stratified_split,
)


def _mide22_frame():
    return pd.DataFrame(
        {
            "topic": ["Covid", "Covid", "Ukraine", "Misc"],
            "event_id": ["EN1", "EN1", "EN2", "EN3"],
            "label": ["False", "True", "Other", "False"],
            "tweet_id": ["1", "2", "3", "4"],
            "text": ["fake claim", "real news", "other", None],
        }
    )


def test_binarize_drops_other_and_maps_labels():
    out = binarize_mide22(_mide22_frame())
    assert out["label"].tolist() == [1, 0, 1]
    assert out["text"].tolist() == ["fake claim", "real news", ""]
    assert list(out.columns) == ["tweet_id", "event_id", "topic", "text", "label"]


def test_binarize_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        binarize_mide22(pd.DataFrame({"label": ["True"]}))


def test_load_mide22_tsv(tmp_path):
    p = tmp_path / "en.tsv"
    _mide22_frame().fillna("").to_csv(p, sep="\t", index=False)
    assert len(load_mide22_tsv(p)) == 3


def test_load_dataset_validates(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"], "label": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="labels must be 0/1"):
        load_dataset(p)
    pd.DataFrame({"tweet": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(p)


def test_split_proportions_and_disjointness(sample_df):
    s = stratified_split(sample_df)
    n = len(sample_df)
    assert len(s.train) + len(s.val) + len(s.test) == n
    assert len(s.test) == pytest.approx(0.2 * n, abs=1)
    assert len(s.val) == pytest.approx(0.1 * n, abs=1)
    # Together the splits are exactly the original rows (no loss, no duplication).
    combined = pd.concat([s.train, s.val, s.test]).sort_values(["text", "label"])
    expected = sample_df.sort_values(["text", "label"])
    assert combined[["text", "label"]].values.tolist() == expected.values.tolist()


def test_split_is_stratified(sample_df):
    s = stratified_split(sample_df)
    overall = sample_df["label"].mean()
    for part in (s.train, s.val, s.test):
        assert part["label"].mean() == pytest.approx(overall, abs=0.05)


def test_split_is_deterministic(sample_df):
    a, b = stratified_split(sample_df, seed=7), stratified_split(sample_df, seed=7)
    pd.testing.assert_frame_equal(a.test, b.test)


@pytest.mark.parametrize(("val", "test"), [(0.0, 0.2), (0.5, 0.5), (0.1, 1.2)])
def test_split_rejects_bad_sizes(sample_df, val, test):
    with pytest.raises(ValueError):
        stratified_split(sample_df, val_size=val, test_size=test)


def test_splits_save_load_roundtrip(sample_splits, tmp_path):
    sample_splits.save(tmp_path)
    stats = json.loads((tmp_path / "split_stats.json").read_text())
    assert stats["test"]["n"] == len(sample_splits.test)
    loaded = Splits.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.val, sample_splits.val[["text", "label"]])
