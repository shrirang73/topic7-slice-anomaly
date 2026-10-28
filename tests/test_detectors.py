"""Checks of the detectors, stream model and calibration against known answers."""
import numpy as np
import pytest
from scipy.stats import norm

from slicead.detectors import CUSUM, EWMA, Shewhart, StreamModel
from slicead.eval import (arl_with_restarts, calibrate_per_stream, calibrate_threshold, first_alarms,
                          summarize_delays)


def siegmund_one_sided(h, k):
    # Siegmund's approximation for the CUSUM sum(z - k) with z ~ N(0, 1), checked in notebook 01
    D, b = -k, h + 1.166
    return (np.exp(-2 * D * b) + 2 * D * b - 1) / (2 * D**2)


def test_shewhart_arl_is_exact_for_iid_streams():
    rng = np.random.default_rng(0)
    S, h = 5, 3.0
    Z = rng.standard_normal((40, 20_000, S))
    p = 1 - (1 - 2 * norm.sf(h)) ** S
    assert arl_with_restarts(Shewhart(), Z, h) == pytest.approx(1 / p, rel=0.05)


def test_two_sided_cusum_arl_matches_theory():
    # two independent one-sided schemes: 1/ARL = 1/ARL+ + 1/ARL-, i.e. half the one-sided ARL
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((40, 20_000, 1))
    arl = arl_with_restarts(CUSUM(k=0.5), Z, 4.0)
    assert arl == pytest.approx(siegmund_one_sided(4.0, 0.5) / 2, rel=0.08)


def test_ewma_is_scaled_like_a_z_score():
    rng = np.random.default_rng(2)
    det = EWMA(lam=0.2)
    det.reset(2000, 1)
    for _ in range(200):
        score, _ = det.update(rng.standard_normal((2000, 1)))
    assert np.std(det.s[:, 0]) / det.norm == pytest.approx(1.0, rel=0.05)


def test_calibration_hits_target():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((40, 10_000, 3))
    h, arl = calibrate_threshold(CUSUM(k=0.5), Z, target_arl=500)
    assert arl == pytest.approx(500, rel=0.02)
    assert arl_with_restarts(CUSUM(k=0.5), Z, h) == pytest.approx(arl)


def test_stream_model_whitens_ar1():
    rng = np.random.default_rng(4)
    phi, R, N = 0.7, 20, 5000
    e = rng.standard_normal((R, N, 1))
    X = np.empty_like(e)
    X[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
    for n in range(1, N):
        X[:, n] = phi * X[:, n - 1] + e[:, n]
    X = 5 + 2 * X
    m = StreamModel().fit(X)
    assert m.phi[0] == pytest.approx(phi, abs=0.01)
    E = m.whiten(X)
    assert E.std() == pytest.approx(1.0, rel=0.02)
    lag1 = (E[:, 1:] * E[:, :-1]).mean() / E.var()
    assert abs(lag1) < 0.02


def test_first_alarm_finds_a_step_and_its_stream():
    rng = np.random.default_rng(5)
    Z = rng.standard_normal((200, 300, 4))
    Z[:, 100:, 2] += 3.0                            # shift in stream 2 from window 100
    t, s = first_alarms(CUSUM(k=0.5), Z, h=8.0)
    after = t >= 100
    assert after.mean() > 0.9
    assert (s[after] == 2).mean() > 0.95
    assert np.median(t[after] - 100 + 1) <= 5


def test_summarize_delays():
    t = np.array([5, 12, 15, -1, 40])               # change at 10, horizon 20 windows
    out = summarize_delays(t, change_window=10, horizon_windows=20, window_s=0.1)
    assert out["false_alarm"] == pytest.approx(0.2)
    assert out["detected"] == pytest.approx(0.5)     # 12 and 15 detected; -1 and 40 missed
    assert out["delay_mean_s"] == pytest.approx((3 + 6 + 20 + 20) / 4 * 0.1)


def test_per_stream_calibration_equalises_streams():
    # stream 0 is heavy-tailed (t, 2 dof); with one common threshold it would set the budget for all
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((40, 10_000, 3))
    Z[:, :, 0] = rng.standard_t(2, size=(40, 10_000)) / 2
    det = Shewhart()
    c, arl = calibrate_per_stream(det, Z, target_arl=300)
    assert arl == pytest.approx(300, rel=0.02)
    assert det.scale[0] > 1.5 * det.scale[1]                 # heavy-tailed stream gets a higher threshold
    _, which = first_alarms(det, Z, c)
    shares = np.bincount(which[which >= 0], minlength=3) / (which >= 0).sum()
    assert shares.min() > 0.2                                 # alarms now spread over all streams


# ---------------------------------------------------------------- coupled detectors

from slicead.detectors import CUSUMBank, MEWMA, candidate_directions
from slicead.eval import calibrate_bank


def test_candidate_directions():
    D = candidate_directions(3)
    assert D.shape == (20, 3)
    assert len({tuple(r) for r in D}) == 20


def test_single_direction_bank_is_a_scalar_cusum():
    # increments z . Sigma^-1 d - 1/2 with d' Sigma^-1 d = 1 are N(-1/2, 1): one-sided CUSUM, k = 1/2
    rng = np.random.default_rng(7)
    Sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    Z = rng.multivariate_normal([0, 0], Sigma, size=(40, 20_000))
    bank = CUSUMBank([[1.0, 0.3]], Sigma)
    assert arl_with_restarts(bank, Z, 4.0) == pytest.approx(siegmund_one_sided(4.0, 0.5), rel=0.08)


def test_mewma_t2_has_chi2_mean():
    rng = np.random.default_rng(8)
    Sigma = np.array([[1.0, 0.7, 0.4], [0.7, 1.0, 0.5], [0.4, 0.5, 1.0]])
    det = MEWMA(Sigma, lam=0.2)
    det.reset(4000, 3)
    for _ in range(100):
        score, _ = det.update(rng.multivariate_normal(np.zeros(3), Sigma, size=4000))
    assert np.mean(score**2) == pytest.approx(3.0, rel=0.05)


def test_bank_beats_per_stream_cusum_on_correlated_data():
    # rho = 0.9, fault in stream 0 only: notebook 02 predicts a large gain for the coupled detector
    rng = np.random.default_rng(9)
    Sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    Zcal = rng.multivariate_normal([0, 0], Sigma, size=(40, 10_000))
    bank = CUSUMBank(candidate_directions(2), Sigma)
    c_bank, _ = calibrate_bank(bank, Zcal, 500)
    per = CUSUM(0.5)
    c_per, _ = calibrate_per_stream(per, Zcal, 500)
    Z = rng.multivariate_normal([0, 0], Sigma, size=(400, 200))
    Z[:, 50:, 0] += 1.0
    t_b, _ = first_alarms(bank, Z, c_bank)
    t_p, _ = first_alarms(per, Z, c_per)
    d_b = np.where(t_b >= 50, t_b - 49, np.nan)
    d_p = np.where(t_p >= 50, t_p - 49, np.nan)
    assert np.nanmean(d_p) / np.nanmean(d_b) > 1.8


# ---------------------------------------------------------------- union (combined) detector

from slicead.detectors import Union
from slicead.eval import calibrate_union


def test_union_alarms_at_first_component_alarm():
    rng = np.random.default_rng(10)
    Z = rng.standard_normal((50, 300, 3))
    Z[:, 100:, 0] += 1.5
    a, b = CUSUM(0.5), CUSUM(0.5)
    union = Union([(a, [0], 6.0), (b, [1, 2], 7.0)], sizes=[1, 2])
    t_u, w_u = first_alarms(union, Z, 1.0)
    t_a, _ = first_alarms(CUSUM(0.5), Z[:, :, [0]], 6.0)
    t_b, _ = first_alarms(CUSUM(0.5), Z[:, :, [1, 2]], 7.0)
    expected = np.where((t_a >= 0) & ((t_b < 0) | (t_a <= t_b)), t_a, t_b)
    np.testing.assert_array_equal(t_u, expected)
    assert union.decode(int(w_u[t_u == t_a][0]))[0] == 0


def test_union_joint_calibration_hits_target_with_equal_shares():
    rng = np.random.default_rng(11)
    Z = rng.standard_normal((40, 10_000, 3))
    union = Union([(CUSUM(0.5), [0], 1.0), (CUSUM(0.5), [1, 2], 1.0)], sizes=[1, 2])
    c, arl = calibrate_union(union, Z, 500)
    assert arl == pytest.approx(500, rel=0.02)
    # each part alone was set to ARL 1000 (half the budget each)
    for det, cols, h in union.parts:
        assert arl_with_restarts(det, Z[:, :, cols], h) == pytest.approx(1000, rel=0.03)


# ---------------------------------------------------------------- long-run covariance and VAR whitening

from slicead.detectors import VARModel, long_run_cov


def _var1(A, n_runs, T, rng):
    X = np.zeros((n_runs, T, A.shape[0]))
    E = rng.standard_normal(X.shape)
    for t in range(1, T):
        X[:, t] = X[:, t - 1] @ A.T + E[:, t]
    return X[:, 200:]                                   # drop the burn-in


def test_long_run_cov_of_ar1():
    # AR(1) with phi = 0.5 and unit innovations: long-run variance 1 / (1 - phi)^2 = 4
    rng = np.random.default_rng(12)
    X = _var1(np.array([[0.5]]), 40, 5000, rng)
    assert long_run_cov(X, L=60)[0, 0] == pytest.approx(4.0, rel=0.05)


def test_var_recovers_coefficients_and_long_run_cov():
    rng = np.random.default_rng(13)
    A = np.array([[0.5, 0.2], [-0.3, 0.4]])
    X = _var1(A, 20, 5000, rng)
    m = VARModel(max_p=5).fit(X)
    assert m.p == 1
    np.testing.assert_allclose(m.A[0], A * np.outer(1 / m.sd, m.sd), atol=0.02)   # A in standardised units
    E = m.whiten(X)[:, m.p:]
    lag1 = np.einsum("rti,rtj->ij", E[:, 1:], E[:, :-1]) / E[:, 1:, 0].size
    assert np.abs(lag1).max() < 0.02                    # residuals are white
    # the VAR's implied long-run covariance matches the direct estimate (both in standardised units)
    Xs = (X - m.mu) / m.sd
    Ainv = np.linalg.inv(m.dc_gain)
    S_e = m.Sigma_e * np.outer(m.sd_e, m.sd_e)
    np.testing.assert_allclose(Ainv @ S_e @ Ainv.T, long_run_cov(Xs, L=60), rtol=0.08)


def test_diagonal_var_ignores_other_streams():
    rng = np.random.default_rng(14)
    X = _var1(np.array([[0.5, 0.4], [0.0, 0.6]]), 10, 4000, rng)
    m = VARModel(max_p=3, diagonal=True).fit(X)
    assert m.A[:, 0, 1] == pytest.approx(np.zeros(m.p))
    assert m.A[:, 1, 0] == pytest.approx(np.zeros(m.p))


def test_var_residual_directions_match_a_step_shift():
    rng = np.random.default_rng(15)
    A = np.array([[0.6, 0.1], [0.2, 0.5]])
    X = _var1(A, 20, 3000, rng)
    m = VARModel(p=1).fit(X)
    mu = np.array([1.0, -0.5])                          # shift in standardised units
    Xf = X[:, :400].copy()
    Xf[:, 100:] += mu * m.sd
    shift = m.whiten(Xf)[:, 150:].mean(axis=(0, 1)) - m.whiten(X[:, :400])[:, 150:].mean(axis=(0, 1))
    np.testing.assert_allclose(shift, m.residual_directions(mu)[0], atol=1e-9)


def test_matched_bank_is_a_cusum_with_reference_half_the_shift():
    # shift 0.6 on one unit-variance stream: increment z - 0.3, i.e. Siegmund with k = 0.3
    rng = np.random.default_rng(16)
    Z = rng.standard_normal((40, 40_000, 1))
    bank = CUSUMBank.matched([0.6], [[1.0]])
    assert bank.k[0] == pytest.approx(0.3)
    assert arl_with_restarts(bank, Z, 6.0) == pytest.approx(siegmund_one_sided(6.0, 0.3), rel=0.08)
    # the Mahalanobis length sets k in coloured noise
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    d = np.array([1.0, 0.0])
    assert CUSUMBank.matched(d, Sigma).k[0] == pytest.approx(0.5 * np.sqrt(d @ np.linalg.solve(Sigma, d)))
    assert CUSUMBank(candidate_directions(2), Sigma).subset([0, 3]).k == pytest.approx([0.5, 0.5])
