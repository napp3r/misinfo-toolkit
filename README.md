# misinfo-toolkit

[![CI](https://github.com/napp3r/misinfo-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/napp3r/misinfo-toolkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A small, tested and reproducible Python toolkit for **tweet-level misinformation detection**.
It packages the classical baselines from the paper *"Fake News Detection on Twitter Using
Machine Learning"* (D. Karatay, IEEE SIST 2026) for the
[MiDe22](https://github.com/ogozcelik/MiDe22) dataset:

* tweet preprocessing (URL / mention removal, hashtag normalisation, tokenisation);
* **causal cue** features: connectives (*because, therefore*), causative verbs (*causes, led to*)
  and "X causes Y" patterns;
* models: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, TF-IDF + causal cues;
* stratified 70/10/20 splits with a fixed seed, evaluation with the *fake* class as positive;
* a `misinfo` command-line tool and machine-readable reports (JSON, CSV, Markdown, PNG).

## Installation

```bash
git clone https://github.com/napp3r/misinfo-toolkit.git
cd misinfo-toolkit
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .            # core library + CLI
pip install -e ".[dev]"     # + pytest, ruff, pre-commit (for contributors)
```

## Quick start (synthetic sample data)

The repository ships `data/sample/sample_tweets.csv` - **240 synthetic, template-generated
tweets** (see `scripts/make_sample_data.py`) with the same schema as the real data.

```bash
misinfo split data/sample/sample_tweets.csv -o data/splits
misinfo evaluate data/splits -o reports          # -> reports/results.{json,csv,md}, metrics.png
misinfo train data/sample/sample_tweets.csv -m logreg_cues -o models/model.joblib
misinfo predict --model models/model.joblib "BREAKING: 5G towers secretly cause panic!!"
```

Or from Python:

```python
from misinfo_toolkit.data import load_dataset, stratified_split
from misinfo_toolkit.evaluate import run_experiment, results_table

splits = stratified_split(load_dataset("data/sample/sample_tweets.csv"))
print(results_table(run_experiment(splits, ["logreg", "svm", "logreg_cues"])))
```

## Using the real MiDe22 data

MiDe22 is published as **tweet IDs only** (Twitter/X terms of service), so tweet texts are
not included in this repository. Obtain the dataset from the
[official repository](https://github.com/ogozcelik/MiDe22), hydrate the tweets so the TSV
has a `text` column, then:

```bash
misinfo prepare path/to/mide22_en_misinfo_tweets.tsv -o data/processed/mide22_en.csv
misinfo split data/processed/mide22_en.csv -o data/splits
misinfo evaluate data/splits -o results/mide22
```

`prepare` keeps the `True`/`False` labels (False -> 1 = misinformation, True -> 0 = truthful)
and drops `Other`. `data/processed`, `data/splits` and `models/` are git-ignored.

## CLI reference

| Command | Purpose |
|---|---|
| `misinfo prepare TSV -o CSV` | Binarise a hydrated MiDe22 TSV into a `text,label` CSV |
| `misinfo split CSV -o DIR [--val-size --test-size --seed]` | Stratified train/val/test split |
| `misinfo evaluate DIR -o OUT [-m logreg svm logreg_cues]` | Train on train, validate, refit on train+val, test, write report |
| `misinfo train CSV -m MODEL -o FILE` | Fit one model on all rows and save it (joblib) |
| `misinfo predict TEXT... --model FILE` | Print a misinformation score and verdict per text |

## Project structure

```
src/misinfo_toolkit/
  preprocess.py   # clean_tweet(), tokenize()
  cues.py         # causal cue features + span finder
  data.py         # MiDe22 loader, binarisation, stratified split
  models.py       # model factory, scoring, save/load
  evaluate.py     # metrics, experiment protocol, reports
  cli.py          # `misinfo` command
tests/            # pytest suite (unit + end-to-end)
data/sample/      # synthetic demo dataset
docs/             # technology choices
.github/          # CI workflow, issue/PR templates, dependabot
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install                     # run ruff on every commit
ruff check . && ruff format --check .  # lint
pytest --cov                           # tests, coverage gate 85 %
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch / commit / PR workflow.

## Continuous integration

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

1. **Lint** - `ruff check` and `ruff format --check`;
2. **Test** - pytest with coverage on Python 3.10 / 3.11 / 3.12 (Ubuntu) and 3.12 on Windows and macOS;
3. **Reproducibility run** - executes the full CLI pipeline on the sample data and publishes the
   results table in the job summary and as an artifact;
4. **Build** - builds the wheel / sdist and validates them with `twine check`.

## Technology choices

Python + scikit-learn + pandas, pytest, ruff, GitHub Actions - the rationale for each choice is
in [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md).

## Citation

If you use this toolkit, please cite it via [CITATION.cff](CITATION.cff) and cite the MiDe22
dataset (Toraman et al., LREC-COLING 2024).

## License

[MIT](LICENSE) © 2026 Dair Karatay
