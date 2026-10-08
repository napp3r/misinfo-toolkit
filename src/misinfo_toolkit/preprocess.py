"""Text normalisation for tweets.

Two branches are provided, mirroring the experimental setup of the paper:

* :func:`clean_tweet` - light cleaning suitable for transformer models
  (URLs and @mentions removed, ``#`` stripped from hashtags, lower-cased).
* :func:`tokenize` - tokenisation + stop-word removal used as the analyzer
  of the TF-IDF vectorizer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WS_RE = re.compile(r"\s+")
# Words with an optional apostrophe part (e.g. "don't"); numbers are dropped.
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_tweet(text: str | None) -> str:
    """Remove URLs and @mentions, keep hashtag words, lower-case and squeeze spaces."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text.lower()).strip()


def tokenize(
    text: str | None,
    *,
    remove_stopwords: bool = True,
    extra_stopwords: Iterable[str] | None = None,
) -> list[str]:
    """Tokenise a tweet for TF-IDF features.

    The text is first passed through :func:`clean_tweet`, then split into
    alphabetic tokens. English stop words (scikit-learn list) are removed
    unless ``remove_stopwords`` is ``False``.
    """
    tokens = _TOKEN_RE.findall(clean_tweet(text))
    if not remove_stopwords:
        return tokens
    stop = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stop |= {w.lower() for w in extra_stopwords}
    return [t for t in tokens if t not in stop]
