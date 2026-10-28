"""False-alarm calibration (ARL) and detection delay for the detectors in slicead.detectors.

All functions take detector input Z of shape (n_runs, n_windows, n_streams) and work in windows.
"""
import numpy as np


def arl_with_restarts(det, Z, h):
    """Average run length to false alarm on fault-free data Z, at threshold h.

    The detector is restarted after every alarm, so a run of N windows with m alarms contributes m
    run lengths; ARL = total windows / total alarms (inf if there are none). For detectors that fully
    reset after an alarm this is the renewal estimate of E[T].
    """
    R, N, S = Z.shape
    det.reset(R, S)
    n_alarms = 0
    for n in range(N):
        score, _ = det.update(Z[:, n])
        hit = np.flatnonzero(score >= h)
        if hit.size:
            n_alarms += hit.size
            det.clear(hit)
    return np.inf if n_alarms == 0 else R * N / n_alarms


def calibrate_threshold(det, Z, target_arl, h_lo=0.0, h_hi=None, tol=0.01, max_iter=60):
    """Threshold h whose simulated ARL on fault-free Z is target_arl (bisection on log ARL).

    Returns (h, achieved ARL). The ARL is non-decreasing in h for a fixed Z, so bisection works.
    """
    if h_hi is None:
        h_hi = 1.0
        while arl_with_restarts(det, Z, h_hi) < target_arl:
            h_hi *= 2
            if h_hi > 1e4:
                raise RuntimeError("could not bracket the target ARL")
    arl = None
    for _ in range(max_iter):
        h = 0.5 * (h_lo + h_hi)
        arl = arl_with_restarts(det, Z, h)
        if abs(np.log(arl / target_arl)) < tol:
            break
        if arl < target_arl:
            h_lo = h
        else:
            h_hi = h
    return h, arl


def first_alarms(det, Z, h):
    """First alarm window of each run (or -1) and the stream that raised it (or -1)."""
    R, N, S = Z.shape
    det.reset(R, S)
    t_alarm = np.full(R, -1)
    stream = np.full(R, -1)
    for n in range(N):
        score, which = det.update(Z[:, n])
        new = (score >= h) & (t_alarm < 0)
        t_alarm[new] = n
        stream[new] = which[new]
        if (t_alarm >= 0).all():
            break
    return t_alarm, stream


def summarize_delays(t_alarm, change_window, horizon_windows, window_s):
    """Delay statistics for one detector on runs with a fault starting at change_window.

    - false_alarm: fraction of runs that alarmed before the change (dropped from the delay);
    - detected: fraction of the remaining runs that alarmed within horizon_windows of the change;
    - delay_mean_s: conditional mean delay in seconds, with misses counted as the horizon
      (a lower bound on the true mean when anything is missed);
    - delay_median_s: median delay over the remaining runs (inf if more than half are missed).
    The delay of an alarm in the first faulty window is one window.
    """
    t = np.asarray(t_alarm)
    fa = (t >= 0) & (t < change_window)
    keep = ~fa
    d = np.where((t >= change_window) & (t < change_window + horizon_windows),
                 t - change_window + 1, np.inf)[keep]
    detected = np.isfinite(d)
    capped = np.minimum(d, horizon_windows)
    return dict(false_alarm=fa.mean(), detected=detected.mean() if keep.any() else np.nan,
                delay_mean_s=capped.mean() * window_s if keep.any() else np.nan,
                delay_median_s=np.median(d) * window_s if keep.any() else np.nan)


def calibrate_per_stream(det, Z, target_arl):
    """Give every stream an equal share of the false-alarm budget, then calibrate the combination.

    Each stream j alone gets the threshold h_j with ARL = n_streams * target_arl; det.scale is set to
    (h_j). A final threshold c (close to 1) is then calibrated for the maximum over streams of
    stat_j / h_j to reach target_arl. Returns (c, achieved ARL); det.scale is left set.
    """
    S = Z.shape[2]
    det.scale = None
    det.scale = np.array([calibrate_threshold(det, Z[:, :, [j]], S * target_arl)[0] for j in range(S)])
    return calibrate_threshold(det, Z, target_arl)


def calibrate_bank(bank, Z, target_arl):
    """Equalised calibration for a CUSUMBank: each candidate alone gets ARL = n_candidates * target,
    then one factor brings the whole bank to target_arl. Returns (c, achieved ARL); bank.scale is set."""
    m = bank.n_components
    bank.scale = None
    bank.scale = np.array([calibrate_threshold(bank.subset(i), Z, m * target_arl)[0] for i in range(m)])
    return calibrate_threshold(bank, Z, target_arl)


def calibrate_union(union, Z, target_arl):
    """Joint calibration of a detectors.Union: each part alone gets ARL = n_parts * target_arl on its
    own columns (an equal share of the false-alarm budget), then one factor c brings the union to
    target_arl. Part thresholds are updated in place; returns (c, achieved ARL)."""
    n = len(union.parts)
    new = []
    for det, cols, _ in union.parts:
        h, _ = calibrate_threshold(det, Z[:, :, cols], n * target_arl)
        new.append((det, cols, h))
    union.parts = new
    return calibrate_threshold(union, Z, target_arl)
