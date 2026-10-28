"""Simulator configuration: dataclasses and YAML loading."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class TrafficConfig:
    """Packet arrival process of one slice.

    "poisson": Poisson arrivals at rate_pps packets per second.
    "onoff":   Markov-modulated Poisson. The source alternates between ON and OFF periods with
               exponential (geometric in slots) durations of mean mean_on_ms / mean_off_ms, and
               sends Poisson traffic at on_rate_pps / off_rate_pps in each state.
    """
    model: str
    rate_pps: float = 0.0
    on_rate_pps: float = 0.0
    off_rate_pps: float = 0.0
    mean_on_ms: float = 0.0
    mean_off_ms: float = 0.0

    def mean_rate_pps(self) -> float:
        if self.model == "poisson":
            return self.rate_pps
        p_on = self.mean_on_ms / (self.mean_on_ms + self.mean_off_ms)
        return p_on * self.on_rate_pps + (1 - p_on) * self.off_rate_pps


@dataclass
class SliceConfig:
    name: str
    weight: float          # share of the server's capacity guaranteed when all slices are busy
    packet_bits: int
    buffer_ms: float       # buffer size, in ms of the slice's guaranteed rate (weight * capacity)
    radio_loss: float      # probability that a transmitted packet is lost on the radio link
    traffic: TrafficConfig


@dataclass
class CapacityConfig:
    """Server capacity C_n = mbps * clip(1 + x_n), with x_n an AR(1) process.

    The fluctuation stands for changing radio conditions (spectral efficiency) that hit every
    slice at once. variation_std is the standard deviation of x_n, variation_ms its correlation time.
    """
    mbps: float
    variation_std: float = 0.0
    variation_ms: float = 1.0
    min_factor: float = 0.3
    max_factor: float = 1.7


@dataclass
class SimConfig:
    slot_ms: float
    window_slots: int      # KPIs are computed per window of this many slots
    warmup_s: float        # simulated, then discarded, so queues start in steady state
    capacity: CapacityConfig
    slices: list[SliceConfig]

    @property
    def window_s(self) -> float:
        return self.window_slots * self.slot_ms / 1000

    def offered_load(self) -> float:
        """Mean offered traffic divided by mean capacity (ignores capacity clipping)."""
        bps = sum(s.traffic.mean_rate_pps() * s.packet_bits for s in self.slices)
        return bps / (self.capacity.mbps * 1e6)


FAULT_KINDS = ("capacity", "traffic", "radio_loss", "weight")


@dataclass
class Fault:
    """A change in one model parameter, starting start_s seconds after warm-up (the change time).

    kind: "capacity" (server capacity), "traffic" (a slice's arrival rates), "radio_loss" (a slice's
          radio loss probability) or "weight" (a slice's scheduling weight; the weights are then
          re-normalised to sum to 1, so the other slices gain what this one loses).
    The parameter is multiplied by `factor`, at once (ramp_s = 0) or linearly over ramp_s seconds.
    """
    kind: str
    start_s: float
    factor: float
    slice: str | None = None
    ramp_s: float = 0.0

    def __post_init__(self):
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if (self.kind == "capacity") != (self.slice is None):
            raise ValueError("capacity faults take no slice; all other faults need one")
        if self.start_s < 0 or self.factor < 0 or self.ramp_s < 0:
            raise ValueError("start_s, factor and ramp_s must be non-negative")

    def multiplier(self, t_s):
        """Factor applied to the parameter at times t_s (seconds after warm-up)."""
        t = np.asarray(t_s, float)
        if self.ramp_s > 0:
            frac = np.clip((t - self.start_s) / self.ramp_s, 0.0, 1.0)
        else:
            frac = (t >= self.start_s).astype(float)
        return 1.0 + (self.factor - 1.0) * frac


def load_fault_scenarios(path, start_s) -> dict[str, list[Fault]]:
    """Named fault scenarios from YAML (name -> list of faults), all starting at start_s."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {name: [Fault(start_s=start_s, **f) for f in faults] for name, faults in raw.items()}


def load_config(path) -> SimConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> SimConfig:
    slices = [SliceConfig(**{**s, "traffic": TrafficConfig(**s["traffic"])}) for s in raw["slices"]]
    for s in slices:
        if s.traffic.model not in ("poisson", "onoff"):
            raise ValueError(f"slice {s.name}: unknown traffic model {s.traffic.model!r}")
        if s.traffic.model == "onoff" and min(s.traffic.mean_on_ms, s.traffic.mean_off_ms) <= raw["slot_ms"]:
            raise ValueError(f"slice {s.name}: on/off periods must be longer than one slot")
    if abs(sum(s.weight for s in slices) - 1) > 1e-6:
        raise ValueError("slice weights must sum to 1")
    return SimConfig(slot_ms=raw["slot_ms"], window_slots=raw["window_slots"], warmup_s=raw["warmup_s"],
                     capacity=CapacityConfig(**raw["capacity"]), slices=slices)
