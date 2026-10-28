"""Checks of the cell-level resource utilisation KPI."""
import numpy as np
import pytest

from slicead.detectors import stream_array
from slicead.sim import Fault, load_config, simulate
from test_faults import mean_after
from test_sim import make_cfg, poisson_slice


def test_saturated_server_is_fully_utilised():
    cfg = make_cfg([poisson_slice("a", 0.5, 200_000, buffer_ms=20.0),
                    poisson_slice("b", 0.5, 200_000, buffer_ms=20.0)], variation_std=0.1)
    res = simulate(cfg, n_windows=200, seed=21)
    np.testing.assert_allclose(res.utilisation, 1.0, rtol=1e-9)


def test_light_load_utilisation_is_offered_over_capacity():
    cfg = make_cfg([poisson_slice("a", 1.0, 30_000)])           # 30 Mbit/s into 100 Mbit/s
    res = simulate(cfg, n_windows=500, n_runs=2, seed=22)
    assert res.utilisation.mean() == pytest.approx(0.30, rel=0.01)
    assert (res.utilisation <= 1 + 1e-12).all() and (res.utilisation >= 0).all()


def test_capacity_drop_raises_utilisation_by_inverse_factor():
    cfg = make_cfg([poisson_slice("a", 1.0, 30_000)])
    res = simulate(cfg, n_windows=400, n_runs=4, seed=23, faults=[Fault("capacity", 20.0, 0.7)])
    before = mean_after(res.utilisation[..., None], res, 0, 20)[0]
    after = mean_after(res.utilisation[..., None], res, 21)[0]
    assert after / before == pytest.approx(1 / 0.7, rel=0.02)


def test_utilisation_balances_throughput_and_capacity():
    cfg = load_config("configs/sim_default.yaml")
    res = simulate(cfg, n_windows=300, n_runs=2, seed=24)
    sent = res.kpi("throughput_mbps").sum(axis=-1)                     # delivered, all slices
    lost = (res.kpi("loss_pct") / 100 * res.offered_mbps).sum(axis=-1)  # upper bound on radio losses
    used = res.utilisation * res.capacity_mbps
    assert (used >= sent - 1e-6).all()
    assert used.mean() == pytest.approx(sent.mean(), rel=0.01)          # radio losses are ~0.1%
    assert (used - sent).mean() <= lost.mean() + 1e-6


def test_stream_array_selects_utilisation():
    cfg = load_config("configs/sim_default.yaml")
    res = simulate(cfg, n_windows=20, seed=25)
    X = stream_array(res, [("eMBB", "latency_ms"), ("cell", "utilisation")])
    np.testing.assert_array_equal(X[:, :, 1], res.utilisation)
