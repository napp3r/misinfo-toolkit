"""Dataset loading, label binarisation and stratified splitting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_SEED

#: MiDe22 label -> binary target (1 = misinformation, 0 = truthful). "Other" is dropped.
LABEL_MAP = {"False": 1, "True": 0}
SPLIT_NAMES = ("train", "val", "test")


def binarize_mide22(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ``True``/``False`` rows of a MiDe22 frame and map them to 0/1."""
    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"MiDe22 frame is missing columns: {sorted(missing)}")
    out = df[df["label"].isin(list(LABEL_MAP))].copy()
    out["label"] = out["label"].map(LABEL_MAP).astype(int)
    out["text"] = out["text"].fillna("").astype(str)
    keep = [c for c in ("tweet_id", "event_id", "topic", "text", "label") if c in out.columns]
    return out[keep].reset_index(drop=True)


def load_mide22_tsv(path: str | Path) -> pd.DataFrame:
    """Read a hydrated MiDe22 TSV (columns incl. ``label`` and ``text``) and binarise it."""
    raw = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, on_bad_lines="skip")
    return binarize_mide22(raw)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a ``text,label`` CSV and validate it."""
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    bad = set(df["label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"{path}: labels must be 0/1, found {sorted(bad)}")
    return df


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def stats(self) -> dict[str, dict[str, int]]:
        return {name: class_stats(getattr(self, name)) for name in SPLIT_NAMES}

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in SPLIT_NAMES:
            getattr(self, name)[["text", "label"]].to_csv(out / f"{name}.csv", index=False)
        (out / "split_stats.json").write_text(json.dumps(self.stats(), indent=2))

    @classmethod
    def load(cls, in_dir: str | Path) -> Splits:
        d = Path(in_dir)
        return cls(*(load_dataset(d / f"{name}.csv") for name in SPLIT_NAMES))


def class_stats(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts()
    return {"n": len(df), "misinfo_1": int(counts.get(1, 0)), "truthful_0": int(counts.get(0, 0))}


def stratified_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = RANDOM_SEED,
) -> Splits:
    """Split into train/val/test (default 70/10/20), stratified by ``label``."""
    if not 0 < test_size < 1 or not 0 < val_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in (0, 1) and sum to < 1")
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    # val_size is a fraction of the whole dataset, so rescale it to train_val.
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=train_val["label"],
    )
    return Splits(*(s.reset_index(drop=True) for s in (train, val, test)))
