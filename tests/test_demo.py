"""The Streamlit demo must keep working: it runs, and its default scenario shows the coupled win."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "demo" / "app.py"
pytestmark = pytest.mark.skipif(not (APP.parent / "calibration.json").exists(),
                                reason="run demo/prepare.py first")


def metrics(at):
    return {m.label: (m.value, m.delta or "") for m in at.metric}


def test_default_scenario_shows_coupled_detector_first():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    m = metrics(at)
    std = float(m["Standard detector (11 KPIs)"][0].split("at ")[1].split(" s")[0])
    cpl = float(m["Coupled detector (latency)"][0].split("at ")[1].split(" s")[0])
    assert 20 < cpl < std                       # both after the fault at 20 s, coupled first


def test_no_fault_default_seed_is_quiet():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].set_value("none")
    at.run()
    assert not at.exception
    assert all(v == "no alarm" for label, (v, _) in metrics(at).items() if "detector" in label)


def alarm_time(value):
    return float(value.split("at ")[1].split(" s")[0])


def test_combined_detector_takes_the_faster_part():
    for fault, part in [("urllc_weight_misconfig", "coupled"), ("urllc_surge", "standard")]:
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        at.sidebar.selectbox[0].set_value(fault)
        at.run()
        assert not at.exception
        m = metrics(at)
        times = [alarm_time(m[k][0]) for k in ("Standard detector (11 KPIs)", "Coupled detector (latency)")]
        comb_value, comb_delta = m["Combined detector"]
        assert alarm_time(comb_value) <= min(times) + 0.15      # within a window or so of the faster part
        assert f"via {part} part" in comb_delta
