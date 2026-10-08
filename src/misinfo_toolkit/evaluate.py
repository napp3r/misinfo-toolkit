"""Metrics, experiment runner and report generation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import RANDOM_SEED
from .data import Splits
from .models import build_model, display_name


@dataclass(frozen=True)
class BinaryMetrics:
    """Metrics with misinformation (label 1) as the positive class."""

    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_fake=float(p),
        recall_fake=float(r),
        f1_fake=float(f1),
    )


def run_experiment(
    splits: Splits,
    model_names: Iterable[str],
    *,
    max_features: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, dict]:
    """Train and evaluate each model with the protocol used in the paper.

    1. fit on ``train`` and report validation metrics;
    2. refit on ``train + val`` and report test metrics.
    """
    train_val = pd.concat([splits.train, splits.val], ignore_index=True)
    results: dict[str, dict] = {}
    for name in model_names:
        model = build_model(name, max_features=max_features, seed=seed)
        model.fit(splits.train["text"], splits.train["label"])
        val_m = compute_metrics(splits.val["label"], model.predict(splits.val["text"]))

        model.fit(train_val["text"], train_val["label"])
        test_pred = model.predict(splits.test["text"])
        test_m = compute_metrics(splits.test["label"], test_pred)
        results[name] = {
            "model": display_name(name),
            "val": val_m.to_dict(),
            "test": test_m.to_dict(),
            "confusion_matrix": confusion_matrix(
                splits.test["label"], test_pred, labels=[0, 1]
            ).tolist(),
        }
    return results


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Tabulate test-set metrics, one row per model."""
    rows = [
        {
            "Model": r["model"],
            "Accuracy": r["test"]["accuracy"],
            "Precision (Fake)": r["test"]["precision_fake"],
            "Recall (Fake)": r["test"]["recall_fake"],
            "F1 (Fake)": r["test"]["f1_fake"],
        }
        for r in results.values()
    ]
    return pd.DataFrame(rows)


def _to_markdown(table: pd.DataFrame) -> str:
    def fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "|" + "|".join("---" for _ in table.columns) + "|",
    ]
    lines += [
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)
    ]
    return "\n".join(lines) + "\n"


def write_report(results: dict[str, dict], out_dir: str | Path) -> list[Path]:
    """Write results.json / .csv / .md and a metrics.png bar chart to ``out_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = results_table(results)

    paths = [out / "results.json", out / "results.csv", out / "results.md", out / "metrics.png"]
    paths[0].write_text(json.dumps(results, indent=2))
    table.to_csv(paths[1], index=False)
    paths[2].write_text(_to_markdown(table))

    metrics = ["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1 (Fake)"]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(table), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in table.iterrows():
        ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["Model"])
    ax.set_xticks(x + width * (len(table) - 1) / 2, metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths[3], dpi=200)
    plt.close(fig)
    return paths
