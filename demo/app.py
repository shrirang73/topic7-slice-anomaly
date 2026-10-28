"""Streamlit demo: live QoS anomaly detection on three simulated 5G slices.

Run from the project root (after `python demo/prepare.py` once):
    .venv\\Scripts\\python -m streamlit run demo/app.py
"""
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from slicead.detectors import CUSUM, CUSUMBank, StreamModel, stream_array
from slicead.sim import Fault, load_config, load_fault_scenarios, simulate

ROOT = Path(__file__).resolve().parents[1]
SLICES = ["eMBB", "URLLC", "mMTC"]
SLICE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
DETECTOR_COLORS = {"standard": "#52514e", "coupled": "#6250d6", "combined": "#c98500"}
FAULT_LABELS = {
    "none": "No fault",
    "urllc_weight_misconfig": "URLLC scheduling weight misconfigured",
    "urllc_surge": "URLLC traffic surge",
    "mmtc_storm": "mMTC device storm",
    "embb_surge": "eMBB traffic surge",
    "capacity_drop": "Cell capacity drop (all slices)",
    "capacity_degradation": "Gradual capacity degradation (all slices)",
    "urllc_loss_burst": "URLLC packet-loss burst",
}
DURATION_S = 60.0

st.set_page_config(page_title="5G slice anomaly detection", layout="wide")


# ----------------------------------------------------------------------------- set-up

@st.cache_resource
def load_detectors():
    path = ROOT / "demo" / "calibration.json"
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    model = StreamModel()
    for k, v in cal["model"].items():
        setattr(model, k, np.array(v))
    standard = CUSUM(cal["standard"]["k"])
    standard.scale = np.array(cal["standard"]["scale"])
    bank = CUSUMBank(cal["coupled"]["directions"], cal["coupled"]["Sigma"])
    bank.scale = np.array(cal["coupled"]["scale"])
    return dict(cal=cal, model=model, streams=[tuple(s) for s in cal["streams"]], lat=cal["latency_index"],
                standard=(standard, cal["standard"]["threshold"]), coupled=(bank, cal["coupled"]["threshold"]),
                combined=cal["combined"])


@st.cache_data(show_spinner="Simulating 60 s of three slices (1 ms resolution)...")
def run_simulation(fault, strength, fault_s, seed):
    cfg = load_config(ROOT / "configs" / "sim_default.yaml")
    faults = []
    if fault != "none":
        base = load_fault_scenarios(ROOT / "configs" / "faults.yaml", start_s=fault_s)[fault]
        faults = [Fault(f.kind, f.start_s, f.factor ** strength, f.slice, f.ramp_s) for f in base]
    res = simulate(cfg, n_windows=int(DURATION_S / cfg.window_s), n_runs=1, seed=seed, faults=faults)
    return dict(time=res.time_s, latency=res.kpi("latency_ms")[0], throughput=res.kpi("throughput_mbps")[0],
                loss=res.kpi("loss_pct")[0], X=stream_array(res, D["streams"]), window_s=res.window_s,
                factor=faults[0].factor if faults else None)


def score_path(det, Z, threshold):
    """Detector score / threshold for every window (no restart), and what it points at."""
    det.reset(1, Z.shape[2])
    score, which = np.empty(Z.shape[1]), np.empty(Z.shape[1], dtype=int)
    for n in range(Z.shape[1]):
        s, w = det.update(Z[:, n])
        score[n], which[n] = s[0] / threshold, w[0]
    return score, which


def blame_slice(Zlat, a, Sigma, lookback=10):
    """Which slice the coupled alarm points to, from the latency residuals of the last second.

    If all three slices moved the same way by comparable amounts, the fault is shared ("all slices").
    Otherwise blame the slice whose latency moved most beyond what the other slices explain: its
    standardised regression residual (Sigma^-1 m)_i / sqrt((Sigma^-1)_ii), where m is the recent mean.
    """
    m = Zlat[0, max(0, a - lookback + 1):a + 1].mean(axis=0)
    big = np.abs(m).max()
    if len(set(np.sign(m))) == 1 and np.abs(m).min() >= 0.5 * big:
        return "all slices " + ("up" if m[0] > 0 else "down")
    Si = np.linalg.inv(Sigma)
    r = Si @ m / np.sqrt(np.diag(Si))
    i = int(np.argmax(np.abs(r)))
    return f"{SLICES[i]} latency " + ("up" if r[i] > 0 else "down")


D = load_detectors()
st.title("Quickest detection of QoS faults across 5G network slices")
if D is None:
    st.error("Calibration not found. Run `.venv\\Scripts\\python demo\\prepare.py` once from the project root.")
    st.stop()

# ----------------------------------------------------------------------------- controls

with st.sidebar:
    st.header("Scenario")
    fault = st.selectbox("Fault", list(FAULT_LABELS), index=1, format_func=FAULT_LABELS.get)
    strength = st.select_slider("Fault strength", options=[0.03, 0.05, 0.1, 0.25, 0.5, 1.0], value=0.1,
                                help="Fraction of the full fault size, on a log scale (1 = the size in "
                                     "configs/faults.yaml). Small strengths are where detectors differ.",
                                disabled=fault == "none")
    fault_s = st.slider("Fault starts at (s)", 10, 40, 20, disabled=fault == "none")
    seed = st.number_input("Random seed", 0, 10_000, 0, step=1,
                           help="Each seed is a different random run. With one false alarm per 100 s, about "
                                "half of all 60 s runs show a false alarm somewhere; seed 0 has none.")
    st.divider()
    st.header("Playback")
    animate = st.toggle("Animate in real time", value=False)
    speed = st.select_slider("Speed", options=[1, 2, 5, 10], value=5, format_func=lambda v: f"{v}x",
                             disabled=not animate)
    st.caption(f"All three detectors are calibrated to one false alarm per "
               f"{D['cal']['target_arl_windows'] * 0.1:.0f} s on average in normal operation.")

sim = run_simulation(fault, strength, float(fault_s), int(seed))
Xs = D["model"].standardize(sim["X"])                  # (1 run, n_windows, 11 streams)
Xw = D["model"].whiten(sim["X"])[:, :, D["lat"]]       # whitened latency of the three slices
std_score, std_which = score_path(D["standard"][0], Xs, D["standard"][1])
cpl_score, cpl_which = score_path(D["coupled"][0], Xw, D["coupled"][1])
# combined detector: alarm when either part reaches its (raised) threshold inside the union
cb = D["combined"]
parts = np.stack([std_score * D["standard"][1] / cb["standard_threshold"],
                  cpl_score * D["coupled"][1] / cb["coupled_threshold"]]) / cb["factor"]
comb_score, comb_part = parts.max(axis=0), parts.argmax(axis=0)
t = sim["time"]
fault_t = fault_s if fault != "none" else None


def first_alarm(score):
    idx = np.flatnonzero(score >= 1)
    return int(idx[0]) if idx.size else None


alarms = {"standard": first_alarm(std_score), "coupled": first_alarm(cpl_score), "combined": first_alarm(comb_score)}

with st.expander("What am I looking at?", expanded=False):
    st.markdown(
        "Three network slices (eMBB video, URLLC low-latency control, mMTC sensors) share one cell, "
        "simulated at 1 ms resolution; KPIs are reported every 100 ms. A fault is injected at the chosen "
        "time. Two detectors watch the KPIs:\n\n"
        "- **Standard**: a CUSUM on each of 11 KPIs (throughput, latency, jitter, loss of every slice) "
        "separately, alarm when any one crosses its threshold.\n"
        "- **Coupled**: a bank of 20 CUSUMs on the three slices' latencies *together*. Each looks for one "
        "fault pattern (one slice alone, all together, pairs) after removing the delay noise the slices "
        "share, so it can see small faults that are hidden in shared noise.\n"
        "- **Combined**: both of the above together; it alarms as soon as either does. To keep the same "
        "false-alarm rate, each part gets half the false-alarm budget, so its thresholds are slightly "
        "higher than when used alone.\n\n"
        "All three raise a false alarm about once every 100 s in normal operation, so their speed can be "
        "compared fairly. Each alarm also names the **symptom** that tripped it, which need not be the "
        "slice where the root cause is: when URLLC's scheduling weight is cut, the capacity it loses "
        "goes to the other slices, so the first visible symptom is often *mMTC getting faster*.\n\n"
        "Try the default (a small URLLC weight misconfiguration), then a URLLC traffic surge, then a "
        "cell capacity drop, a fault shared by all slices, where coupling does not help.")

# ----------------------------------------------------------------------------- display


def draw(upto):
    """Render everything up to window index `upto` (exclusive)."""
    tt = t[:upto]
    m1, m2, m3, m4 = metrics.columns(4)
    if fault_t is None:
        m1.metric("Fault", "none")
    else:
        m1.metric("Fault", FAULT_LABELS[fault].split(" (")[0],
                  f"x{sim['factor']:.3g} at {fault_t:.0f} s" if tt[-1] >= fault_t else "not yet started",
                  delta_color="off")
    for col, key, name in [(m2, "standard", "Standard detector (11 KPIs)"),
                           (m3, "coupled", "Coupled detector (latency)"),
                           (m4, "combined", "Combined detector")]:
        a = alarms[key]
        if a is None or a >= upto:
            col.metric(name, "no alarm")
            continue
        when = t[a] + sim["window_s"]
        if fault_t is not None and when > fault_t:
            label = f"delay {when - fault_t:.1f} s"
        else:
            label = "false alarm" + (" (before the fault)" if fault_t is not None else "")
        part = key if key != "combined" else ("standard", "coupled")[comb_part[a]]
        if part == "standard":
            s, k = D["streams"][std_which[a]]
            culprit = f"{s} {k.split('_')[0]}"
        else:
            culprit = blame_slice(Xw, a, D["coupled"][0].Sigma)
        if key == "combined":
            culprit += f" (via {part} part)"
        col.metric(name, f"ALARM at {when:.1f} s", f"{label} · symptom: {culprit}", delta_color="off")

    rules = []
    if fault_t is not None:
        rules.append(dict(time=fault_t, what="fault starts"))
    for key, name in [("standard", "standard alarm"), ("coupled", "coupled alarm"), ("combined", "combined alarm")]:
        if alarms[key] is not None and alarms[key] < upto:
            rules.append(dict(time=t[alarms[key]] + sim["window_s"], what=name))
    rule_df = pd.DataFrame(rules or [dict(time=np.nan, what="")])
    rule_layer = alt.Chart(rule_df).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(
        x="time:Q", color=alt.Color("what:N", scale=alt.Scale(
            domain=["fault starts", "standard alarm", "coupled alarm", "combined alarm"],
            range=["#e34948", DETECTOR_COLORS["standard"], DETECTOR_COLORS["coupled"], DETECTOR_COLORS["combined"]]),
            legend=alt.Legend(title=None, orient="top")))
    xscale = alt.Scale(domain=[0, DURATION_S])

    cols = charts.columns(3)
    for i, (col, name) in enumerate(zip(cols, SLICES)):
        df = pd.DataFrame({"time": tt, "latency (ms)": sim["latency"][:upto, i],
                           "throughput (Mbit/s)": sim["throughput"][:upto, i]})
        base = alt.Chart(df).encode(x=alt.X("time:Q", scale=xscale, title="time (s)"))
        lat = base.mark_line(color=SLICE_COLORS[i], strokeWidth=1.3).encode(
            y=alt.Y("latency (ms):Q", scale=alt.Scale(zero=False)))
        thr = base.mark_line(color=SLICE_COLORS[i], strokeWidth=1.3, opacity=0.8).encode(
            y=alt.Y("throughput (Mbit/s):Q", scale=alt.Scale(zero=False)))
        col.markdown(f"**{name}**")
        col.altair_chart((lat + rule_layer).properties(height=170), width="stretch")
        col.altair_chart((thr + rule_layer).properties(height=140), width="stretch")

    sdf = pd.DataFrame({"time": np.concatenate([tt, tt, tt]),
                        "score / threshold": np.concatenate([std_score[:upto], cpl_score[:upto], comb_score[:upto]]),
                        "detector": (["standard (11 KPIs)"] * len(tt) + ["coupled (latency)"] * len(tt)
                                     + ["combined"] * len(tt))})
    score_chart = alt.Chart(sdf).mark_line(strokeWidth=2).encode(
        x=alt.X("time:Q", scale=xscale, title="time (s)"),
        y=alt.Y("score / threshold:Q", scale=alt.Scale(type="symlog"), title="score / threshold"),
        color=alt.Color("detector:N", scale=alt.Scale(
            domain=["standard (11 KPIs)", "coupled (latency)", "combined"],
            range=[DETECTOR_COLORS["standard"], DETECTOR_COLORS["coupled"], DETECTOR_COLORS["combined"]]),
            legend=alt.Legend(title=None, orient="top")))
    one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#e34948", strokeWidth=1.5).encode(y="y:Q")
    detector_box.markdown("**Detector statistics** (an alarm is raised when a curve reaches 1)")
    detector_box.altair_chart((score_chart + one).properties(height=220), width="stretch")


if not animate:
    upto_s = st.slider("Show up to (s)", 1.0, DURATION_S, DURATION_S, step=0.5)
metrics = st.container()
charts = st.container()
detector_box = st.container()

if animate:
    placeholder = st.empty()
    step = max(1, int(speed))                     # windows per frame; one frame per 0.1 s of wall time
    for upto in range(step, len(t) + 1, step):
        with placeholder.container():
            metrics = st.container()
            charts = st.container()
            detector_box = st.container()
            draw(upto)
        time.sleep(0.1)
else:
    draw(int(round(upto_s / sim["window_s"])))
