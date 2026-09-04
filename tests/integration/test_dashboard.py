from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_locked_portfolio_without_exceptions() -> None:
    script = Path(__file__).parents[2] / "scripts" / "run_dashboard.py"
    app = AppTest.from_file(script, default_timeout=300).run()
    assert not app.exception
    assert app.title[0].value == "ReturnGuard"
    assert any(metric.label == "Requests" and metric.value == "2,400" for metric in app.metric)
