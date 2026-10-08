from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_and_classifies_example():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.selectbox(key="example").select("Conspiracy-style claim").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == "Likely misinformation"
    assert any("Causal cues found" in s.value for s in at.subheader)


def test_highlight_escapes_html():
    import importlib.util

    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    # Only the pure helper is needed; executing the module outside `streamlit run` is fine.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.highlight_cues("<b>x</b> because y")
    assert out == "&lt;b&gt;x&lt;/b&gt; <mark>because</mark> y"
