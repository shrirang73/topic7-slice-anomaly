"""Turning simulator output into detector input: stream arrays, standardisation and AR(1) whitening."""
import numpy as np


def stream_array(res, streams):
    """KPI streams of a SimResult as an array (n_runs, n_windows, n_streams).

    streams: list of (slice_name, kpi_name); ("cell", "utilisation") selects the cell-level resource
    utilisation. Windows with an undefined KPI (no arrivals) carry the previous value forward (or the
    next one, at the very start).
    """
    cols = [res.utilisation if (s, k) == ("cell", "utilisation") else res.kpi(k)[:, :, res.slice_names.index(s)]
            for s, k in streams]
    X = np.stack(cols, axis=-1).astype(float)
    for r in range(X.shape[0]):
        for j in range(X.shape[2]):
            x = X[r, :, j]
            bad = np.isnan(x)
            if bad.any() and not bad.all():
                idx = np.where(~bad, np.arange(len(x)), 0)
                np.maximum.accumulate(idx, out=idx)
                x[:] = x[idx]
                first = np.argmax(~bad)
                x[:first] = x[first]
    return X


class StreamModel:
    """Fault-free model of each stream: mean, standard deviation and AR(1) coefficient.

    Fitted once on fault-free training runs, then applied unchanged to new data, as a deployed
    detector would be. `standardize` gives z-scores; `whiten` gives the standardised residuals of the
    AR(1) prediction-error filter e_n = (x_n - mu) - phi (x_{n-1} - mu).
    """

    def fit(self, X):
        """X: fault-free data (n_runs, n_windows, n_streams)."""
        self.mu = X.mean(axis=(0, 1))
        self.sd = X.std(axis=(0, 1))
        Xc = X - self.mu
        num = (Xc[:, 1:] * Xc[:, :-1]).sum(axis=(0, 1))
        den = (Xc[:, :-1] ** 2).sum(axis=(0, 1))
        self.phi = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        E = Xc[:, 1:] - self.phi * Xc[:, :-1]
        self.sd_e = E.std(axis=(0, 1))
        return self

    def standardize(self, X):
        return (X - self.mu) / np.where(self.sd > 0, self.sd, 1.0)

    def whiten(self, X):
        Xc = X - self.mu
        E = np.empty_like(Xc)
        E[:, 0] = Xc[:, 0] * np.sqrt(np.clip(1 - self.phi**2, 0, None))   # stationary start
        E[:, 1:] = Xc[:, 1:] - self.phi * Xc[:, :-1]
        return E / np.where(self.sd_e > 0, self.sd_e, 1.0)


def long_run_cov(Z, L=50):
    """Long-run covariance of a stationary multivariate series: the Bartlett-weighted sum of
    cross-covariances, sum_l (1 - |l|/(L+1)) Gamma(l) for |l| <= L. It is 2 pi times the cross-spectral
    density at frequency zero, the noise covariance a sustained (DC) fault competes with.

    Z: (n_runs, n_windows, n_streams); runs are treated as independent pieces of one process.
    """
    Zc = Z - Z.mean(axis=(0, 1))
    S = np.einsum("rti,rtj->ij", Zc, Zc) / (Zc.shape[0] * Zc.shape[1])
    for lag in range(1, L + 1):
        G = np.einsum("rti,rtj->ij", Zc[:, lag:], Zc[:, :-lag]) / (Zc.shape[0] * (Zc.shape[1] - lag))
        S = S + (1 - lag / (L + 1)) * (G + G.T)
    return S


def _lagged(X, p):
    """Design matrix of p lags, without crossing run boundaries: (n_runs * (T - p), n_streams * p)."""
    n, T, m = X.shape
    return np.concatenate([X[:, p - l:T - l] for l in range(1, p + 1)], axis=2).reshape(-1, m * p)


class VARModel:
    """Fault-free VAR(p) model of standardised streams, fitted by least squares, order by BIC.

    `diagonal=True` fits a separate AR(p) to each stream (no cross-stream terms): the richer per-stream
    whitening. Otherwise the streams are whitened jointly (vector autoregression). `whiten` returns the
    one-step prediction errors divided by their sd; the first p windows of each run are set to 0.
    A sustained shift mu of the standardised streams appears in the residuals, once the filter has
    settled, as `dc_gain @ mu` (divided by sd_e), with dc_gain = I - sum_l A_l.
    """

    def __init__(self, max_p=30, diagonal=False, p=None):
        self.max_p, self.diagonal, self.p_fixed = max_p, diagonal, p

    def _fit_p(self, X, p):
        Y = X[:, p:].reshape(-1, X.shape[2])
        D = _lagged(X, p)
        m = X.shape[2]
        if self.diagonal:
            B = np.zeros((m * p, m))
            for j in range(m):
                cols = [l * m + j for l in range(p)]
                B[cols, j] = np.linalg.lstsq(D[:, cols], Y[:, j], rcond=None)[0]
            k = p * m
        else:
            B = np.linalg.lstsq(D, Y, rcond=None)[0]
            k = p * m * m
        E = Y - D @ B
        S = E.T @ E / len(E)
        bic = len(E) * np.log(np.linalg.det(S)) + k * np.log(len(E))
        return B, S, bic

    def fit(self, X):
        """X: fault-free data (n_runs, n_windows, n_streams), in raw units."""
        self.mu = X.mean(axis=(0, 1))
        self.sd = X.std(axis=(0, 1))
        Xs = (X - self.mu) / np.where(self.sd > 0, self.sd, 1.0)
        orders = [self.p_fixed] if self.p_fixed else range(1, self.max_p + 1)
        fits = {p: self._fit_p(Xs, p) for p in orders}
        self.bic = {p: f[2] for p, f in fits.items()}
        self.p = min(self.bic, key=self.bic.get)
        B, S, _ = fits[self.p]
        m = X.shape[2]
        self.A = B.T.reshape(m, self.p, m).transpose(1, 0, 2)    # A[l][i, j]: effect of x_j(t-l-1) on x_i(t)
        self.sd_e = np.sqrt(np.diag(S))
        self.Sigma_e = S / np.outer(self.sd_e, self.sd_e)        # correlation of the standardised residuals
        self.dc_gain = np.eye(m) - self.A.sum(axis=0)
        return self

    def whiten(self, X):
        Xs = (X - self.mu) / np.where(self.sd > 0, self.sd, 1.0)
        n, T, m = Xs.shape
        E = np.zeros_like(Xs)
        pred = sum(Xs[:, self.p - l - 1:T - l - 1] @ self.A[l].T for l in range(self.p))
        E[:, self.p:] = (Xs[:, self.p:] - pred) / self.sd_e
        return E

    def residual_directions(self, D):
        """Map shift directions of the standardised streams to their steady-state residual shifts."""
        return (np.atleast_2d(D) @ self.dc_gain.T) / self.sd_e
