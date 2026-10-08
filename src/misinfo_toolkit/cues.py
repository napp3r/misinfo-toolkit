"""Hand-crafted causal cue features.

Misinformation often relies on simplified causal claims ("X causes Y",
"this led to ..."). Each feature is a binary indicator of a causal
connective, a causative verb or a short "X causes Y" pattern.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalCueSpec:
    connectives: tuple[str, ...]
    causative_verbs: tuple[str, ...]
    patterns: tuple[tuple[str, str], ...]  # (feature name, regex)

    def feature_names(self) -> list[str]:
        return (
            [f"conn__{c}" for c in self.connectives]
            + [f"verb__{v}" for v in self.causative_verbs]
            + [name for name, _ in self.patterns]
        )


DEFAULT_CUES = CausalCueSpec(
    connectives=(
        "because",
        "therefore",
        "thus",
        "hence",
        "due to",
        "as a result",
        "consequently",
        "since",
        "so that",
        "resulting in",
    ),
    causative_verbs=(
        "cause",
        "causes",
        "caused",
        "causing",
        "lead to",
        "leads to",
        "led to",
        "result in",
        "results in",
        "resulted in",
        "trigger",
        "triggers",
        "triggered",
        "create",
        "creates",
        "created",
    ),
    patterns=(
        ("pattern_x_causes_y", r"\b\w+(?:\s+\w+){0,3}\s+causes\s+\w+"),
        ("pattern_x_led_to_y", r"\b\w+(?:\s+\w+){0,3}\s+led\s+to\s+\w+"),
    ),
)


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(phrase)}\b")


def _compile(spec: CausalCueSpec) -> list[re.Pattern[str]]:
    regexes = [_phrase_regex(p) for p in (*spec.connectives, *spec.causative_verbs)]
    regexes += [re.compile(p) for _, p in spec.patterns]
    return regexes


def extract_cues(text: str | None, spec: CausalCueSpec = DEFAULT_CUES) -> np.ndarray:
    """Return a binary (0/1) ``float32`` vector of cue indicators for one text."""
    t = (text or "").lower()
    return np.array([r.search(t) is not None for r in _compile(spec)], dtype=np.float32)


def extract_cue_matrix(texts: Iterable[str], spec: CausalCueSpec = DEFAULT_CUES) -> np.ndarray:
    """Vectorised version of :func:`extract_cues` - shape ``(n_texts, n_features)``."""
    regexes = _compile(spec)
    rows = [[r.search((t or "").lower()) is not None for r in regexes] for t in texts]
    return np.array(rows, dtype=np.float32).reshape(len(rows), len(regexes))


def find_cue_spans(text: str, spec: CausalCueSpec = DEFAULT_CUES) -> list[tuple[int, int, str]]:
    """Locate connectives / causative verbs in ``text`` (used for highlighting in the UI).

    Returns non-overlapping ``(start, end, phrase)`` tuples sorted by position;
    longer phrases win over shorter ones that overlap them.
    """
    lowered = text.lower()
    phrases = sorted((*spec.connectives, *spec.causative_verbs), key=len, reverse=True)
    spans: list[tuple[int, int, str]] = []
    for phrase in phrases:
        for m in _phrase_regex(phrase).finditer(lowered):
            if all(m.end() <= s or m.start() >= e for s, e, _ in spans):
                spans.append((m.start(), m.end(), phrase))
    return sorted(spans)
