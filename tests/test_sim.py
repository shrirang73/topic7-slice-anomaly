"""Checks of the slice simulator against cases with known answers."""
import numpy as np
import pytest

from slicead.sim import load_config, simulate, waterfill
from slicead.sim.config import config_from_dict


def make_cfg(slices, mbps=100.0, variation_std=0.0, window_slots=100, warmup_s=1.0):
    return config_from_dict({
        "slot_ms": 1.0, "window_slots": window_slots, "warmup_s": warmup_s,
        "capacity": {"mbps": mbps, "variation_std": variation_std, "variation_ms": 100.0},
        "slices": slices,
    })


def poisson_slice(name, weight, rate_pps, packet_bits=1000, buffer_ms=1000.0, radio_loss=0.0):
    return {"name": name, "weight": weight, "packet_bits": packet_bits, "buffer_ms": buffer_ms,
            "radio_loss": radio_loss, "traffic": {"model": "poisson", "rate_pps": rate_pps}}


# ---------------------------------------------------------------- water-filling

W3 = np.array([0.5, 0.3, 0.2])


@pytest.mark.parametrize("demand, expected", [
    ([1e9, 1e9, 1e9], [50, 30, 20]),                       # all saturated: shares follow weights
    ([1e9, 0, 1e9], [100 * 0.5 / 0.7, 0, 100 * 0.2 / 0.7]), # idle slice's share is passed on
    ([1e9, 5, 1e9], [95 * 5 / 7, 5, 95 * 2 / 7]),          # light slice takes what it needs
    ([10, 5, 3], [10, 5, 3]),                              # everything fits
])
def test_waterfill(demand, expected):
    alloc = waterfill(np.array([demand], float), W3, np.array([100.0]))
    np.testing.assert_allclose(alloc[0], expected, rtol=1e-9)


def test_waterfill_is_work_conserving():
    rng = np.random.default_rng(0)
    demand = rng.exponential(40, size=(1000, 3)) * (rng.random((1000, 3)) < 0.7)
    cap = rng.uniform(20, 150, size=1000)
    alloc = waterfill(demand, W3, cap)
    assert (alloc <= demand + 1e-9).all() and (alloc >= -1e-12).all()
    np.testing.assert_allclose(alloc.sum(axis=1), np.minimum(cap, demand.sum(axis=1)), rtol=1e-9)


# ---------------------------------------------------------------- whole simulator

def test_light_load_latency_matches_theory():
    # One slice alone, Poisson N ~ Poi(lam) packets of L bits per slot, capacity C >> arrivals, so the
    # queue empties every slot. The middle bit of a slot's batch a = N*L waits (a/2)/C slots, and the
    # bit-weighted mean is E[a^2]/(2 E[a] C) = L (1 + lam) / (2 C).
    lam, L, C = 5.0, 1000, 1e5
    cfg = make_cfg([poisson_slice("only", 1.0, lam * 1000, packet_bits=L)])
    res = simulate(cfg, n_windows=2000, n_runs=2, seed=1)
    theory_ms = L * (1 + lam) / (2 * C)
    assert np.nanmean(res.kpi("latency_ms")) == pytest.approx(theory_ms, rel=0.02)
    assert np.nanmean(res.kpi("loss_pct")) == 0.0


def test_saturated_slices_share_by_weight():
    cfg = make_cfg([poisson_slice("a", 0.7, 200_000, buffer_ms=20.0),
                    poisson_slice("b", 0.3, 200_000, buffer_ms=20.0)])
    res = simulate(cfg, n_windows=300, seed=2)
    thr = res.kpi("throughput_mbps").mean(axis=(0, 1))
    np.testing.assert_allclose(thr, [70, 30], rtol=0.01)
    assert (res.kpi("loss_pct").mean(axis=(0, 1)) > 50).all()   # both offered 200 Mbit/s


def test_unused_share_is_passed_on():
    # slice a has weight 0.3 (30 Mbit/s guaranteed) but is offered 80 Mbit/s; b uses only 5 of its 70.
    cfg = make_cfg([poisson_slice("a", 0.3, 80_000), poisson_slice("b", 0.7, 5_000)])
    res = simulate(cfg, n_windows=600, seed=3)
    thr = res.kpi("throughput_mbps").mean(axis=(0, 1))
    np.testing.assert_allclose(thr, [80, 5], rtol=0.01)
    assert res.kpi("loss_pct").max() == 0.0


def test_throughput_balances_offered_minus_losses():
    cfg = load_config("configs/sim_default.yaml")
    res = simulate(cfg, n_windows=600, n_runs=2, seed=4)
    offered = res.offered_mbps.mean(axis=(0, 1))
    thr = res.kpi("throughput_mbps").mean(axis=(0, 1))
    loss = res.kpi("loss_pct").mean(axis=(0, 1)) / 100
    np.testing.assert_allclose(thr, offered * (1 - loss), rtol=0.01)


def test_delay_bounded_by_buffer():
    # overloaded slice: a full buffer of b bits drains at >= its guaranteed rate, so no bit waits
    # longer than buffer_ms / min_capacity_factor (+1 slot for the slot it departs in)
    cfg = make_cfg([poisson_slice("a", 0.5, 150_000, buffer_ms=20.0), poisson_slice("b", 0.5, 60_000)],
                   variation_std=0.2)
    res = simulate(cfg, n_windows=300, seed=5)
    assert res.kpi("loss_pct")[..., 0].mean() > 1
    assert np.nanmax(res.kpi("latency_ms")[..., 0]) <= 20.0 / cfg.capacity.min_factor + 1


def test_reproducible():
    cfg = load_config("configs/sim_default.yaml")
    a = simulate(cfg, n_windows=50, seed=7)
    b = simulate(cfg, n_windows=50, seed=7)
    c = simulate(cfg, n_windows=50, seed=8)
    np.testing.assert_array_equal(a.kpis, b.kpis)
    assert not np.allclose(a.kpis, c.kpis, equal_nan=True)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        make_cfg([poisson_slice("a", 0.5, 1000), poisson_slice("b", 0.4, 1000)])
