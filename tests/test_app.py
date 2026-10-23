from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_runs_an_optimisation():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    at.selectbox(key="case").set_value("cantilever")
    at.run()
    at.slider(key="nelx_cantilever").set_value(30)
    at.slider(key="maxiter").set_value(20)
    at.button[0].click()
    at.run()
    assert not at.exception
    assert "result" in at.session_state
    assert at.session_state["result"].iterations == 20
    assert len(at.metric) >= 4
