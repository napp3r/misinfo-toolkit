# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-08

### Added
- Tweet preprocessing (`clean_tweet`, `tokenize`) and causal cue feature extraction.
- MiDe22 loader with binary labels, stratified 70/10/20 split.
- TF-IDF + Logistic Regression, TF-IDF + Linear SVM and TF-IDF + causal cues models.
- `misinfo` CLI: `prepare`, `split`, `evaluate`, `train`, `predict`.
- Streamlit demo app.
- GitHub Actions CI (lint, test matrix, reproducibility run, build) and release workflow.
