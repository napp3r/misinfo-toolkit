from pathlib import Path

import pandas as pd
import pytest

from misinfo_toolkit.data import load_dataset, stratified_split

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "sample_tweets.csv"


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return load_dataset(SAMPLE_CSV)


@pytest.fixture(scope="session")
def sample_splits(sample_df):
    return stratified_split(sample_df)
