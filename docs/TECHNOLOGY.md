# Technology choices

This document justifies the main technical decisions of **misinfo-toolkit**.
Each choice was evaluated against the same criteria: suitability for the research task,
ecosystem maturity, reproducibility, cost of CI, learning curve and licence.

| Area | Choice | Alternatives considered | Why |
|---|---|---|---|
| Language | **Python 3.10+** | R, Julia, Java | De-facto standard for NLP/ML research; the original experiments were already in Python; huge library ecosystem; readable for reviewers. 3.10 is the oldest version with modern typing syntax (`X \| Y`) still receiving security fixes. |
| ML library | **scikit-learn** | PyTorch, TensorFlow | TF-IDF + linear models are the strongest baselines on MiDe22 (SVM: 0.8415 accuracy, above fine-tuned BERT). scikit-learn pipelines are deterministic with a fixed seed, CPU-only, and install in seconds - a 2 GB PyTorch stack would make every CI run slow. |
| Data handling | **pandas / NumPy** | polars, plain csv | Standard tabular API, read MiDe22 TSV directly, familiar to every data scientist. Dataset size (~2.5k tweets) does not justify a faster engine. |
| Visualisation | **matplotlib** | seaborn, plotly | Publication-quality static figures (PNG/PDF) for papers, headless rendering in CI (`Agg` backend), no browser needed. |
| Persistence | **joblib** | pickle, ONNX | Recommended by scikit-learn for pipelines with NumPy arrays; one line to save/load. |
| CLI | **argparse** (stdlib) | click, typer | Zero extra dependencies; sub-commands map 1:1 onto pipeline stages (`prepare -> split -> evaluate`). |
| Demo UI | **Streamlit** (optional extra) | Gradio, Flask + JS | Pure-Python UI in < 100 lines; first-class testing API (`streamlit.testing.AppTest`); free hosting on Streamlit Community Cloud. Kept as an optional extra so the core library stays light. |
| Packaging | **pyproject.toml + setuptools** | setup.py, Poetry | PEP 517/621 standard, no extra tool required, `pip install -e .` works everywhere. |
| Testing | **pytest + pytest-cov** | unittest | Fixtures, parametrisation, readable asserts; coverage gate (>= 85 %) enforced in CI. |
| Code quality | **ruff** + **pre-commit** | flake8 + black + isort | One fast tool replaces linter, import sorter and formatter; the same hooks run locally and in CI. |
| VCS & hosting | **Git + GitHub** | GitLab, Bitbucket | Free public hosting, issues, pull requests, releases and Actions in one place; standard platform for open research code. |
| CI/CD | **GitHub Actions** | GitLab CI, Jenkins, CircleCI | Native to the hosting platform, free for public repositories, Linux/Windows/macOS runners, YAML in the repository (pipeline as code), no server to maintain unlike Jenkins. |

## Scope decisions

* **No BERT in the package.** The fine-tuned BERT baseline from the paper needs `torch` and
  `transformers` (> 2 GB) and a GPU for reasonable training time. Its results are reported in
  the README for comparison, but it is intentionally excluded from the toolkit and its CI.
* **No raw tweets in the repository.** MiDe22 is distributed as tweet IDs to comply with the
  Twitter/X terms. The repository ships a synthetic dataset with the same schema; real data is
  loaded from a local, git-ignored path.
