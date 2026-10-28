"""Online detectors over several streams at once.

Each detector keeps a state per run and stream, consumes one window of z-values (n_runs, n_streams)
at a time, and returns a score per run (the largest per-stream statistic) and the stream that gave
it. An alarm is raised when the score reaches a threshold h; `clear` resets the runs that alarmed.
None of these detectors knows the fault's shape or direction: each stream is monitored both ways.

Every detector has an optional per-stream `scale`: each stream's statistic is divided by it before the
maximum is taken. With scale = each stream's own threshold (see eval.calibrate_per_stream), all
streams share the false-alarm budget equally instead of the heaviest-tailed stream setting it.
"""
import numpy as np


def _max_over_streams(stat, scale):
    a = stat if scale is None else stat / scale
    return a.max(axis=1), a.argmax(axis=1)


class Shewhart:
    """Static threshold on the current value: score = max_j |z_j|."""
    name = "Shewhart"
    scale = None

    def reset(self, n_runs, n_streams):
        pass

    def update(self, z):
        return _max_over_streams(np.abs(z), self.scale)

    def clear(self, runs):
        pass


class EWMA:
    """Exponentially weighted moving average S_n = (1 - lam) S_{n-1} + lam z_n per stream.

    The score is max_j |S_j| divided by the stationary sd of S for white unit-variance input,
    sqrt(lam / (2 - lam)), so thresholds are in the same units as a z-score.
    """
    name = "EWMA"

    def __init__(self, lam=0.2):
        self.lam = lam
        self.norm = np.sqrt(lam / (2 - lam))
        self.scale = None

    def reset(self, n_runs, n_streams):
        self.s = np.zeros((n_runs, n_streams))

    def update(self, z):
        self.s = (1 - self.lam) * self.s + self.lam * z
        return _max_over_streams(np.abs(self.s) / self.norm, self.scale)

    def clear(self, runs):
        self.s[runs] = 0.0


class CUSUM:
    """Two-sided CUSUM per stream with reference value k (= half the design shift, in sd units).

    W+_n = max(0, W+_{n-1} + z_n - k) and W-_n = max(0, W-_{n-1} - z_n - k); score = max over
    streams and both directions. For Gaussian z this is the log-likelihood-ratio CUSUM for a shift of
    2k, scaled by 1/(2k).
    """
    name = "CUSUM"

    def __init__(self, k=0.5):
        self.k = k
        self.scale = None

    def reset(self, n_runs, n_streams):
        self.up = np.zeros((n_runs, n_streams))
        self.dn = np.zeros((n_runs, n_streams))

    def update(self, z):
        self.up = np.maximum(0.0, self.up + z - self.k)
        self.dn = np.maximum(0.0, self.dn - z - self.k)
        return _max_over_streams(np.maximum(self.up, self.dn), self.scale)

    def clear(self, runs):
        self.up[runs] = 0.0
        self.dn[runs] = 0.0


class CUSUMBank:
    """Bank of coupled (multivariate) CUSUMs, one per candidate fault direction.

    Streams are assumed standardised with fault-free correlation matrix Sigma. For each candidate
    direction d (scaled so that d' Sigma^-1 d = 1, i.e. unit Mahalanobis length), the increment is the
    log-likelihood ratio of a mean shift d against no shift:  z . Sigma^-1 d - 1/2. That is the matched
    filter in coloured noise (whiten, then correlate), and each CUSUM has KL = 1/2 per sample, like
    CUSUM(k=0.5) on one stream. The score is the maximum over candidates (optionally divided by a
    per-candidate `scale`), and the "stream" returned is the index of the winning candidate.

    `k` (default 1/2, scalar or one per candidate) is the reference value subtracted from each
    increment. With k = |d|_M / 2 for a candidate d of Mahalanobis length |d|_M, the increment is the
    exact log-likelihood ratio for a shift d divided by |d|_M: a CUSUM matched to the fault's size.
    """
    name = "CUSUM bank"

    def __init__(self, directions, Sigma, k=0.5):
        D = np.atleast_2d(np.asarray(directions, float))
        Si = np.linalg.inv(np.asarray(Sigma, float))
        norms = np.sqrt(np.einsum("ij,jk,ik->i", D, Si, D))
        self.directions = D / norms[:, None]
        self.A = Si @ self.directions.T                     # (n_streams, n_candidates)
        self.Sigma = np.asarray(Sigma, float)
        self.k = np.broadcast_to(np.asarray(k, float), (D.shape[0],)).copy()
        self.scale = None

    @classmethod
    def matched(cls, shift, Sigma):
        """A one-candidate CUSUM matched to a known shift: direction and reference k = |shift|_M / 2."""
        shift = np.asarray(shift, float)
        return cls([shift], Sigma, k=0.5 * np.sqrt(shift @ np.linalg.solve(np.asarray(Sigma, float), shift)))

    def subset(self, idx):
        """A bank with only the candidates in idx (used to give each candidate its own threshold)."""
        idx = np.atleast_1d(idx)
        return CUSUMBank(self.directions[idx], self.Sigma, k=self.k[idx])

    @property
    def n_components(self):
        return self.A.shape[1]

    def reset(self, n_runs, n_streams):
        self.W = np.zeros((n_runs, self.A.shape[1]))

    def update(self, z):
        self.W = np.maximum(0.0, self.W + z @ self.A - self.k)
        return _max_over_streams(self.W, self.scale)

    def clear(self, runs):
        self.W[runs] = 0.0


class MEWMA:
    """Multivariate EWMA (Hotelling T^2 of the smoothed vector), a fully shape-agnostic chart.

    S_n = (1 - lam) S_{n-1} + lam z_n; score = sqrt(S' Sigma_S^-1 S) with the stationary covariance
    Sigma_S = lam / (2 - lam) * Sigma for white input with correlation Sigma.
    """
    name = "MEWMA"

    def __init__(self, Sigma, lam=0.2):
        self.lam = lam
        self.Si = np.linalg.inv(np.asarray(Sigma, float) * lam / (2 - lam))
        self.scale = None

    def reset(self, n_runs, n_streams):
        self.s = np.zeros((n_runs, n_streams))

    def update(self, z):
        self.s = (1 - self.lam) * self.s + self.lam * z
        t = np.sqrt(np.einsum("ij,jk,ik->i", self.s, self.Si, self.s))[:, None]
        return _max_over_streams(t, self.scale)

    def clear(self, runs):
        self.s[runs] = 0.0


def candidate_directions(n_streams):
    """Candidate fault directions for a bank: each stream alone, all together, each pair together and
    each pair in opposite directions, every one with both signs."""
    I = np.eye(n_streams)
    cands = [I[i] for i in range(n_streams)] + [np.ones(n_streams)]
    for i in range(n_streams):
        for j in range(i + 1, n_streams):
            cands += [I[i] + I[j], I[i] - I[j]]
    cands = np.array(cands)
    return np.vstack([cands, -cands])


class Union:
    """Several detectors, each on its own columns of one input; alarm when any reaches its threshold.

    parts: list of (detector, columns, threshold). The score is max_k score_k / threshold_k, so the
    union alarms at 1 (times a final calibration factor). `which` encodes (part, inner index) as
    offset_k + inner, see `decode`.
    """
    name = "union"

    def __init__(self, parts, sizes):
        self.parts = [(det, np.asarray(cols), float(h)) for det, cols, h in parts]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        self.scale = None

    def decode(self, which):
        k = int(np.searchsorted(self.offsets, which, side="right") - 1)
        return k, int(which - self.offsets[k])

    def reset(self, n_runs, n_streams):
        for det, cols, _ in self.parts:
            det.reset(n_runs, len(cols))

    def update(self, z):
        scores, whiches = [], []
        for (det, cols, h), off in zip(self.parts, self.offsets):
            s, w = det.update(z[:, cols])
            scores.append(s / h)
            whiches.append(w + off)
        S, Wh = np.stack(scores, axis=1), np.stack(whiches, axis=1)
        k = S.argmax(axis=1)
        r = np.arange(S.shape[0])
        return S[r, k], Wh[r, k]

    def clear(self, runs):
        for det, _, _ in self.parts:
            det.clear(runs)
