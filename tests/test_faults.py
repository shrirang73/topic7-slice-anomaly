"""Checks of fault injection and labels."""
import numpy as np
import pytest

from slicead.sim import Fault, load_config, load_fault_scenarios, simulate
from test_sim import make_cfg, poisson_slice


def mean_after(x, res, t0_s, t1_s=None):
    """Mean of x (n_runs, n_windows, ...) over windows lying entirely in [t0_s, t1_s)."""
    t = res.time_s
    t1_s = t[-1] + res.window_s if t1_s is None else t1_s
    sel = (t >= t0_s) & (t + res.window_s <= t1_s)
    return x[:, sel].mean(axis=(0, 1))


def test_unit_factor_fault_changes_nothing():
    cfg = load_config("configs/sim_default.yaml")
    base = simulate(cfg, n_windows=100, n_runs=2, seed=11)
    faults = [Fault("capacity", 2.0, 1.0), Fault("traffic", 2.0, 1.0, slice="URLLC"),
              Fault("radio_loss", 2.0, 1.0, slice="mMTC"), Fault("weight", 2.0, 1.0, slice="eMBB")]
    same = simulate(cfg, n_windows=100, n_runs=2, seed=11, faults=faults)
    np.testing.assert_array_equal(base.kpis, same.kpis)


def test_labels_and_fault_window():
    cfg = make_cfg([poisson_slice("a", 1.0, 10_000)])
    res = simulate(cfg, n_windows=100, faults=[Fault("traffic", 4.0, 2.0, slice="a")])
    assert res.fault_start_s == 4.0 and res.fault_window == 40
    assert not res.labels[:40].any() and res.labels[40:].all()
    assert not simulate(cfg, n_windows=10).labels.any()
    # float rounding must not label the window that ends exactly at the change (59.9 + 0.1 > 60.0)
    res = simulate(cfg, n_windows=700, faults=[Fault("traffic", 60.0, 2.0, slice="a")])
    assert res.fault_window == 600 and not res.labels[599]


def test_capacity_fault_scales_saturated_throughput():
    cfg = make_cfg([poisson_slice("a", 0.5, 200_000, buffer_ms=20.0),
                    poisson_slice("b", 0.5, 200_000, buffer_ms=20.0)])
    res = simulate(cfg, n_windows=200, seed=12, faults=[Fault("capacity", 10.0, 0.5)])
    thr = res.kpi("throughput_mbps").sum(axis=-1, keepdims=True)
    assert mean_after(thr, res, 0, 10)[0] == pytest.approx(100, rel=0.01)
    assert mean_after(thr, res, 11)[0] == pytest.approx(50, rel=0.01)


def test_traffic_fault_scales_offered_rate():
    cfg = make_cfg([poisson_slice("a", 0.5, 20_000), poisson_slice("b", 0.5, 20_000)])
    res = simulate(cfg, n_windows=200, n_runs=4, seed=13, faults=[Fault("traffic", 10.0, 3.0, slice="a")])
    before, after = mean_after(res.offered_mbps, res, 0, 10), mean_after(res.offered_mbps, res, 10)
    np.testing.assert_allclose(after / before, [3.0, 1.0], rtol=0.02)


def test_traffic_ramp_reaches_half_way_at_mid_ramp():
    cfg = make_cfg([poisson_slice("a", 1.0, 50_000)])
    res = simulate(cfg, n_windows=400, n_runs=4, seed=14,
                   faults=[Fault("traffic", 10.0, 3.0, slice="a", ramp_s=20.0)])
    base = mean_after(res.offered_mbps, res, 0, 10)[0]
    assert mean_after(res.offered_mbps, res, 19.5, 20.5)[0] / base == pytest.approx(2.0, rel=0.03)
    assert mean_after(res.offered_mbps, res, 31)[0] / base == pytest.approx(3.0, rel=0.02)


def test_radio_loss_fault_sets_loss_rate():
    cfg = make_cfg([poisson_slice("a", 1.0, 20_000, radio_loss=1e-4)])
    res = simulate(cfg, n_windows=300, n_runs=4, seed=15, faults=[Fault("radio_loss", 10.0, 200.0, slice="a")])
    assert mean_after(res.kpi("loss_pct"), res, 10)[0] == pytest.approx(2.0, rel=0.05)


def test_weight_fault_shifts_saturated_shares():
    cfg = make_cfg([poisson_slice("a", 0.5, 200_000, buffer_ms=20.0),
                    poisson_slice("b", 0.5, 200_000, buffer_ms=20.0)])
    res = simulate(cfg, n_windows=200, seed=16, faults=[Fault("weight", 10.0, 1 / 3, slice="a")])
    np.testing.assert_allclose(mean_after(res.kpi("throughput_mbps"), res, 11), [25, 75], rtol=0.01)


def test_fault_validation():
    with pytest.raises(ValueError):
        Fault("capacity", 1.0, 0.5, slice="a")
    with pytest.raises(ValueError):
        Fault("traffic", 1.0, 2.0)
    with pytest.raises(ValueError):
        Fault("jitter", 1.0, 2.0, slice="a")
    cfg = make_cfg([poisson_slice("a", 1.0, 1000)])
    with pytest.raises(ValueError):
        simulate(cfg, n_windows=10, faults=[Fault("traffic", 0.5, 2.0, slice="nope")])


def test_fault_scenarios_load():
    scen = load_fault_scenarios("configs/faults.yaml", start_s=60.0)
    assert len(scen) == 7 and all(f.start_s == 60.0 for fs in scen.values() for f in fs)
