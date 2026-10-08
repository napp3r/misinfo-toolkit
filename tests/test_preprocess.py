import pytest

from misinfo_toolkit.preprocess import clean_tweet, tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Check https://t.co/abc123 now", "check now"),
        ("see www.example.com", "see"),
        ("@WHO says hi", "says hi"),
        ("#Covid19 is   HERE", "covid19 is here"),
        ("  multiple\n\tspaces  ", "multiple spaces"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_tweet(raw, expected):
    assert clean_tweet(raw) == expected


def test_tokenize_removes_stopwords_and_numbers():
    assert tokenize("The vaccine is NOT safe in 2021!") == ["vaccine", "safe"]


def test_tokenize_keeps_apostrophes():
    assert "don't" in tokenize("I don't trust it", remove_stopwords=False)


def test_tokenize_extra_stopwords():
    assert tokenize("vaccine news today", extra_stopwords=["News"]) == ["vaccine", "today"]


def test_tokenize_strips_urls_mentions_hashtags():
    assert tokenize("@bob read https://x.co/1 #FakeNews") == ["read", "fakenews"]
