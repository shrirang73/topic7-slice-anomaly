"""Slotted simulation of network slices that share one server.

Time runs in slots of `slot_ms`. In every slot:

1. Each slice's packets arrive (Poisson or on-off) and join its FIFO queue. Bits that do not fit in
   the buffer are dropped (tail drop).
2. The slot's capacity is shared by work-conserving weighted water-filling: each backlogged slice
   gets capacity in proportion to its weight, and capacity a slice cannot use is passed on to the
   others. This is generalised processor sharing (GPS), applied one slot at a time.
3. A small fraction of the transmitted packets is lost on the radio link.

The FIFO delay of a bit is the horizontal distance between the slice's cumulative arrival and
departure curves. Within a slot, a slice's bits are assumed to leave at its GPS rate given the slices
that are busy in that slot (or faster, if water-filling gave it more than that). KPIs are computed
per window of `window_slots` slots.

Faults (`Fault`) multiply the capacity, or a slice's traffic rate, radio loss or weight, from a change
time on, either as a step or as a linear ramp.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from slicead.sim.config import SimConfig

KPI_NAMES = ("throughput_mbps", "latency_ms", "jitter_ms", "loss_pct")
_EPS = 1e-9


def waterfill(demand, weights, capacity):
    """Share `capacity` (n_runs,) among queues with backlog `demand` (n_runs, n_slices).

    Every backlogged slice gets capacity in proportion to its weight; capacity a slice cannot use
    is re-shared among the others. Returns the allocation, shape (n_runs, n_slices).
    """
    alloc = np.zeros_like(demand)
    left = np.asarray(capacity, dtype=float).copy()
    active = demand > _EPS
    for _ in range(demand.shape[1]):     # each round either fills a slice's demand or uses up the capacity
        wa = weights * active
        total = wa.sum(axis=1)
        per_weight = np.divide(left, total, out=np.zeros_like(left), where=total > 0)
        give = np.minimum(per_weight[:, None] * wa, demand - alloc)
        alloc += give
        left -= give.sum(axis=1)
        active &= demand - alloc > _EPS
    return alloc


@dataclass
class SimResult:
    kpis: np.ndarray            # (n_runs, n_windows, n_slices, n_kpis), KPI order as in kpi_names
    offered_mbps: np.ndarray    # (n_runs, n_windows, n_slices): traffic offered to each slice
    capacity_mbps: np.ndarray   # (n_runs, n_windows): mean server capacity in each window
    utilisation: np.ndarray     # (n_runs, n_windows): bits sent / bits the server could send, all slices
    slice_names: list
    window_s: float
    kpi_names: tuple = KPI_NAMES
    fault_start_s: float | None = None     # change time (earliest fault), or None without faults

    @property
    def time_s(self):
        return np.arange(self.kpis.shape[1]) * self.window_s

    @property
    def labels(self):
        """True for windows that overlap the fault, shape (n_windows,)."""
        if self.fault_start_s is None:
            return np.zeros(self.kpis.shape[1], dtype=bool)
        # window k ends at (k+1) * window_s; the tolerance keeps float rounding (59.9 + 0.1 > 60)
        # from labelling the window that ends exactly at the change time
        return self.time_s + self.window_s > self.fault_start_s + 1e-6 * self.window_s

    @property
    def fault_window(self):
        """Index of the first faulty window, or None without faults."""
        return None if self.fault_start_s is None else int(np.argmax(self.labels))

    def kpi(self, name):
        """One KPI for all runs, windows and slices: shape (n_runs, n_windows, n_slices)."""
        return self.kpis[..., self.kpi_names.index(name)]

    def frame(self, run=0) -> pd.DataFrame:
        """KPIs of one run as a DataFrame indexed by time, with (slice, kpi) columns."""
        cols = pd.MultiIndex.from_product([self.slice_names, self.kpi_names], names=["slice", "kpi"])
        data = self.kpis[run].reshape(self.kpis.shape[1], -1)
        return pd.DataFrame(data, index=pd.Index(self.time_s, name="time_s"), columns=cols)


class _Model:
    """Per-slice constants derived from the config, and the simulation state carried across blocks."""

    def __init__(self, cfg: SimConfig, n_runs: int, rng, faults=(), warm_slots=0):
        tr = [s.traffic for s in cfg.slices]
        slot_s = cfg.slot_ms / 1000
        self.cfg, self.rng, self.R, self.S = cfg, rng, n_runs, len(cfg.slices)
        self.faults, self.warm_slots, self.slot = list(faults), warm_slots, 0
        self.slice_index = {s.name: i for i, s in enumerate(cfg.slices)}
        for f in self.faults:
            if f.slice is not None and f.slice not in self.slice_index:
                raise ValueError(f"fault targets unknown slice {f.slice!r}")
        self.weights = np.array([s.weight for s in cfg.slices], float)
        self.pkt = np.array([s.packet_bits for s in cfg.slices], float)
        self.radio_loss = np.array([s.radio_loss for s in cfg.slices], float)
        self.cap_slot = cfg.capacity.mbps * 1e6 * slot_s                       # bits per slot, nominal
        self.bmax = self.weights * self.cap_slot * np.array([s.buffer_ms for s in cfg.slices]) / cfg.slot_ms
        is_onoff = [t.model == "onoff" for t in tr]
        self.any_onoff = any(is_onoff)
        self.rate_on = np.array([(t.on_rate_pps if o else t.rate_pps) * slot_s for t, o in zip(tr, is_onoff)])
        self.rate_off = np.array([(t.off_rate_pps if o else t.rate_pps) * slot_s for t, o in zip(tr, is_onoff)])
        self.p_leave_on = np.array([cfg.slot_ms / t.mean_on_ms if o else 0.0 for t, o in zip(tr, is_onoff)])
        self.p_leave_off = np.array([cfg.slot_ms / t.mean_off_ms if o else 0.0 for t, o in zip(tr, is_onoff)])
        p_on = np.array([t.mean_on_ms / (t.mean_on_ms + t.mean_off_ms) if o else 1.0 for t, o in zip(tr, is_onoff)])
        cc = cfg.capacity
        self.phi = np.exp(-cfg.slot_ms / cc.variation_ms)
        self.ar_gain = cc.variation_std * np.sqrt(1 - self.phi**2)
        # state, started in the stationary law of the traffic and capacity processes, with empty queues
        self.on = rng.random((n_runs, self.S)) < p_on
        self.x = rng.standard_normal(n_runs) * cc.variation_std
        self.Q = np.zeros((n_runs, self.S))
        self.A = np.zeros((n_runs, self.S))      # cumulative admitted bits
        self.D = np.zeros((n_runs, self.S))      # cumulative departed bits

    def _multipliers(self, B):
        """Fault multipliers for the next B slots: capacity (B,), and traffic, radio_loss and
        weight (B, S). An entry is None when no fault touches that parameter."""
        t_s = (self.slot + np.arange(B) - self.warm_slots) * self.cfg.slot_ms / 1000
        mult = dict.fromkeys(("capacity", "traffic", "radio_loss", "weight"))
        for f in self.faults:
            m = f.multiplier(t_s)
            if f.kind == "capacity":
                mult["capacity"] = m if mult["capacity"] is None else mult["capacity"] * m
            else:
                if mult[f.kind] is None:
                    mult[f.kind] = np.ones((B, self.S))
                mult[f.kind][:, self.slice_index[f.slice]] *= m
        return mult

    def run_block(self, B):
        """Simulate B slots. Returns per-slot arrays plus the cumulative counters at block start."""
        R, S, rng, cc = self.R, self.S, self.rng, self.cfg.capacity
        out = {"A_start": self.A.copy(), "D_start": self.D.copy()}
        mult = self._multipliers(B)
        lam_on, lam_off = self.rate_on, self.rate_off
        if mult["traffic"] is not None:
            lam_on = self.rate_on * mult["traffic"][:, None, :]
            lam_off = self.rate_off * mult["traffic"][:, None, :]
        weights = None
        if mult["weight"] is not None:
            weights = self.weights * mult["weight"]
            weights = weights / weights.sum(axis=1, keepdims=True)
        u_flip = rng.random((B, R, S))
        n_on = rng.poisson(lam_on, size=(B, R, S))
        n_off = rng.poisson(lam_off, size=(B, R, S))
        x, _ = lfilter([self.ar_gain], [1.0, -self.phi], rng.standard_normal((B, R)), axis=0,
                       zi=(self.phi * self.x)[None, :])
        self.x = x[-1]
        cap = self.cap_slot * np.clip(1 + x, cc.min_factor, cc.max_factor)     # (B, R)
        if mult["capacity"] is not None:
            cap = cap * mult["capacity"][:, None]

        offered, admitted, served, rate = (np.empty((B, R, S)) for _ in range(4))
        on, Q = self.on, self.Q
        for t in range(B):
            w_t = self.weights if weights is None else weights[t]
            if self.any_onoff:
                on = on ^ (u_flip[t] < np.where(on, self.p_leave_on, self.p_leave_off))
            off_t = np.where(on, n_on[t], n_off[t]) * self.pkt
            adm = np.minimum(off_t, np.maximum(self.bmax - Q, 0.0))
            Q = Q + adm
            srv = waterfill(Q, w_t, cap[t])
            wa = w_t * (Q > _EPS)
            gps_rate = cap[t][:, None] * wa / np.maximum(wa.sum(axis=1, keepdims=True), _EPS)
            rate[t] = np.maximum(gps_rate, srv)
            Q = np.maximum(Q - srv, 0.0)
            offered[t], admitted[t], served[t] = off_t, adm, srv
        self.on, self.Q = on, Q
        self.A = self.A + admitted.sum(axis=0)
        self.D = self.D + served.sum(axis=0)
        n_tx = np.rint(served / self.pkt).astype(np.int64)
        p_loss = self.radio_loss
        if mult["radio_loss"] is not None:
            p_loss = np.minimum(self.radio_loss * mult["radio_loss"][:, None, :], 1.0)
        lost = np.minimum(rng.binomial(n_tx, p_loss) * self.pkt, served)
        self.slot += B
        out.update(offered=offered, admitted=admitted, served=served, rate=rate, lost=lost, cap=cap)
        return out


def _delays_slots(cur, nxt):
    """FIFO delay (in slots) of the middle bit of each slot's admitted arrivals in block `cur`.

    Uses departures from `cur` and the following block `nxt` as look-ahead. NaN where nothing arrived.
    """
    admitted = cur["admitted"]
    B, R, S = admitted.shape
    D = np.cumsum(np.concatenate([cur["served"], nxt["served"]]), axis=0)   # departures since block start
    rate = np.concatenate([cur["rate"], nxt["rate"]])
    # position of the middle bit on the departure curve, relative to the departures at block start
    target = (cur["A_start"] - cur["D_start"]) + np.cumsum(admitted, axis=0) - admitted / 2
    delay = np.full((B, R, S), np.nan)
    for r in range(R):
        for s in range(S):
            t = np.flatnonzero(admitted[:, r, s] > 0)
            if t.size == 0:
                continue
            tg = target[t, r, s]
            m = np.searchsorted(D[:, r, s], tg, side="left")     # first slot whose departures reach the bit
            if (m >= 2 * B).any():
                raise RuntimeError("delay look-ahead too short: increase block_windows")
            d_before = np.where(m > 0, D[np.maximum(m - 1, 0), r, s], 0.0)
            t_dep = m + (tg - d_before) / rate[m, r, s]
            delay[t, r, s] = t_dep - t
    return delay


def _window_kpis(cur, delay_slots, cfg: SimConfig):
    W, slot_ms = cfg.window_slots, cfg.slot_ms
    B, R, S = cur["admitted"].shape
    nW = B // W
    win = lambda a: a.reshape(nW, W, *a.shape[1:])
    offered, admitted, served, lost = (win(cur[k]) for k in ("offered", "admitted", "served", "lost"))
    d_ms = win(delay_slots) * slot_ms
    window_s = W * slot_ms / 1000

    throughput = (served - lost).sum(axis=1) / window_s / 1e6
    wsum = admitted.sum(axis=1)
    d0 = np.nan_to_num(d_ms)
    with np.errstate(invalid="ignore", divide="ignore"):
        latency = (admitted * d0).sum(axis=1) / wsum
        jitter = np.sqrt((admitted * (d0 - latency[:, None]) ** 2).sum(axis=1) / wsum)
        offered_sum = offered.sum(axis=1)
        loss = np.where(offered_sum > 0,
                        100 * ((offered - admitted).sum(axis=1) + lost.sum(axis=1)) / offered_sum, 0.0)
    kpis = np.stack([throughput, latency, jitter, loss], axis=-1)          # (nW, R, S, 4)
    offered_mbps = offered_sum / window_s / 1e6
    capacity_mbps = win(cur["cap"]).mean(axis=1) / (slot_ms / 1000) / 1e6
    # cell-level resource utilisation, the analogue of a RAN's PRB utilisation: transmitted bits (lost
    # ones included, they used the resource) over the bits the server could have sent in the window
    utilisation = served.sum(axis=(1, 3)) / win(cur["cap"]).sum(axis=1)
    return (kpis.transpose(1, 0, 2, 3), offered_mbps.transpose(1, 0, 2), capacity_mbps.T, utilisation.T)


def simulate(cfg: SimConfig, n_windows: int, n_runs: int = 1, seed: int = 0, block_windows: int = 50,
             faults=()) -> SimResult:
    """Simulate `n_runs` independent replications and return `n_windows` KPI windows of each,
    after discarding `cfg.warmup_s` of warm-up. All runs are advanced together as numpy arrays.

    `faults` (a list of `Fault`) change model parameters from their start time, measured in seconds
    after warm-up; the result's `labels` mark the windows that overlap a fault.
    """
    rng = np.random.default_rng(seed)
    warm = int(np.ceil(cfg.warmup_s / cfg.window_s))
    model = _Model(cfg, n_runs, rng, faults=faults, warm_slots=warm * cfg.window_slots)
    B = block_windows * cfg.window_slots
    n_blocks = -(-(warm + n_windows) // block_windows)

    parts, prev = [], None
    for _ in range(n_blocks + 1):            # the last block only provides look-ahead for delays
        blk = model.run_block(B)
        if prev is not None:
            parts.append(_window_kpis(prev, _delays_slots(prev, blk), cfg))
        prev = blk
    kpis, offered, cap, util = (np.concatenate([p[i] for p in parts], axis=1) for i in range(4))
    keep = slice(warm, warm + n_windows)
    return SimResult(kpis=kpis[:, keep], offered_mbps=offered[:, keep], capacity_mbps=cap[:, keep],
                     utilisation=util[:, keep],
                     slice_names=[s.name for s in cfg.slices], window_s=cfg.window_s,
                     fault_start_s=min((f.start_s for f in faults), default=None))
