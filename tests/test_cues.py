import numpy as np

from misinfo_toolkit.cues import DEFAULT_CUES, extract_cue_matrix, extract_cues, find_cue_spans

NAMES = DEFAULT_CUES.feature_names()


def _active(vec):
    return {NAMES[i] for i in np.flatnonzero(vec)}


def test_feature_vector_shape_and_dtype():
    v = extract_cues("nothing here")
    assert v.shape == (len(NAMES),)
    assert v.dtype == np.float32
    assert v.sum() == 0


def test_detects_connectives_and_verbs():
    active = _active(extract_cues("Prices rose because the war caused shortages"))
    assert {"conn__because", "verb__caused"} <= active


def test_multiword_phrase_and_pattern():
    active = _active(extract_cues("The lockdown led to protests"))
    assert {"verb__led to", "pattern_x_led_to_y"} <= active


def test_word_boundaries_are_respected():
    # "becausewhy" / "recreated" must not trigger "because" / "created"
    assert extract_cues("becausewhy recreated").sum() == 0


def test_case_insensitive_and_none():
    assert extract_cues("BECAUSE").sum() == 1
    assert extract_cues(None).sum() == 0


def test_matrix_matches_single_rows():
    texts = ["this causes that", "", "as a result of x"]
    m = extract_cue_matrix(texts)
    assert m.shape == (3, len(NAMES))
    for row, t in zip(m, texts, strict=True):
        np.testing.assert_array_equal(row, extract_cues(t))


def test_empty_matrix_shape():
    assert extract_cue_matrix([]).shape == (0, len(NAMES))


def test_find_cue_spans_prefers_longest_match():
    text = "Smoking leads to cancer because of tar"
    spans = find_cue_spans(text)
    assert [p for _, _, p in spans] == ["leads to", "because"]
    s, e, _ = spans[0]
    assert text[s:e] == "leads to"
