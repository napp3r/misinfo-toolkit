import json

import pandas as pd
import pytest

from misinfo_toolkit import __version__
from misinfo_toolkit.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_end_to_end_pipeline(tmp_path, sample_csv, capsys):
    splits, reports = tmp_path / "splits", tmp_path / "reports"
    assert main(["split", str(sample_csv), "-o", str(splits)]) == 0
    assert (splits / "train.csv").exists()

    assert main(["evaluate", str(splits), "-o", str(reports), "-m", "logreg", "svm"]) == 0
    results = json.loads((reports / "results.json").read_text())
    assert set(results) == {"logreg", "svm"}
    assert (reports / "metrics.png").stat().st_size > 0
    assert "| Model |" in (reports / "results.md").read_text()

    model = tmp_path / "model.joblib"
    assert main(["train", str(sample_csv), "-o", str(model)]) == 0
    capsys.readouterr()
    assert main(["predict", "--model", str(model), "5G towers secretly causes panic!!"]) == 0
    assert "MISINFORMATION" in capsys.readouterr().out


def test_prepare(tmp_path):
    tsv = tmp_path / "en.tsv"
    pd.DataFrame({"label": ["False", "True", "Other"], "text": ["a", "b", "c"]}).to_csv(
        tsv, sep="\t", index=False
    )
    out = tmp_path / "out" / "data.csv"
    assert main(["prepare", str(tsv), "-o", str(out)]) == 0
    assert pd.read_csv(out)["label"].tolist() == [1, 0]


def test_missing_file_returns_error(tmp_path, capsys):
    assert main(["split", str(tmp_path / "nope.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_unknown_model_choice_is_rejected():
    with pytest.raises(SystemExit):
        main(["train", "x.csv", "-m", "bert"])
