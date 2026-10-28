# Project reference: quickest detection of QoS violations across 5G network slices

*Self-contained briefing on the project as of 28 Oct 2026. It is written so that a person or another LLM can pick the project up without the conversation that produced it. Every number below comes from a notebook in the repository. Items marked **unverified** or **open** have not been established.*

---

## 0. How to use this document (instructions for an LLM reading it)

- **Treat the numbers here as results from our own simulator**, not as facts about real 5G networks. Everything is simulation-only so far.
- **Do not invent papers, results or APIs.** Section 12 lists the references and how far each was checked. If you need something not listed, say so rather than guessing.
- **The project has corrected itself twice. Use the corrected claims (Sections 7 and 9), not the earlier ones:**
  1. the "heavy-tail cancellation" explanation was refuted;
  2. the "zero-frequency rule predicts the gain" claim was refined to a finite-ARL version.
- **The owner (Shrirang Bhale, Math + EEE, BITS Pilani Goa) wants:**
  - explanations linked to signals and systems, probability and detection theory;
  - maths checked numerically;
  - small runnable steps and reproducibility;
  - to be asked before big design decisions.

---

## 1. The project at a glance

- **Context:**
  - 5G hackathon, Topic 7: "Real-Time QoS Deviation & Anomaly Detection Across 5G Network Slices";
  - Communication Systems course, BITS Goa; team of 3 (Shrirang, Palash, Naveen); showcase in the week of 23 Nov 2026.
- **Constraints:**
  - laptop only, no hardware (no SDR, no testbed), Windows 11, Python;
  - executability matters more than ambition.
- **Goals:**
  1. an executable project;
  2. a possible workshop or student-track paper (novelty must be checked, not assumed).
- **Working title:** *Quickest detection of QoS violations in 5G network slices with shared resources.*
- **Core idea in one paragraph.**
  - Network slices (eMBB, URLLC, mMTC) are virtual lanes sharing one physical resource pool, which behaves like weighted fair queueing (generalised processor sharing).
  - So the slices' KPIs are correlated, and a fault in one slice can affect the others.
  - Most anomaly detection watches each KPI separately.
  - We ask when a *coupled* detector, which models the cross-slice correlation, detects faults faster at the *same* false-alarm rate, and why.
- **Headline finding so far.**
  - The useful coupling between slices lives at **low frequency**: slices share slow noise from traffic bursts and capacity drift.
  - A detector that whitens the slices *jointly* (a vector autoregression) and runs a CUSUM on the result detects small slice-local faults **6–33× faster** than per-slice CUSUM in every simulated setting with shared slow noise.
  - Instantaneous correlation, the usual way to reason about coupling, underestimates this by about an order of magnitude.
  - The size of the gain is predicted by detection theory: zero-frequency information rates converted into delays at the operating false-alarm rate.
  - A pre-registered experiment confirmed that the gain grows with the false-alarm interval as predicted.

---

## 2. Background maths (what the reader needs)

### 2.1 Quickest change detection and CUSUM

- **The problem.** A stream $x_1, x_2, \dots$ changes distribution at an unknown time $\nu$. Raise an alarm as soon as possible after $\nu$, with few false alarms.
- **CUSUM** (Page): $W_n = \max(0, W_{n-1} + \ell_n)$, where $\ell_n$ is the log-likelihood ratio of post-change vs pre-change, or a stand-in such as $z_n - k$. Alarm when $W_n \ge h$.
  - For a Gaussian mean shift $\mu$ in unit-variance noise, the matched choice is $k = \mu/2$.
- **The false-alarm metric is the ARL (average run length) to false alarm.** We report it in 100 ms windows; ARL 1000 windows = 100 s.
  - We calibrate thresholds by **simulation** on fault-free runs, with restarts after each alarm, because asymptotic formulas are badly off on real-looking KPIs.
- **The delay metric is the conditional delay:** the mean over runs with no false alarm before the change. A miss counts as the horizon (30 s or 60 s).
- **First-order theory:** delay ≈ log(ARL)/KL, where KL is the information per sample.
- **Siegmund's approximation** gives the ARL and delay of a CUSUM with $N(D,1)$ increments, $E[T] \approx (e^{-2Db} + 2Db - 1)/(2D^2)$ with $b = h + 1.166$. It was checked against Monte Carlo in notebook 01, and we use it for finite-ARL predictions.

### 2.2 The coupled-detector gain

- **Setting:** a sustained mean shift $d$ across slices, with noise covariance $S$.
- **Rates:** the coupled detector's information rate is $\tfrac12 d^\top S^{-1} d$; the best single slice's is $\max_j \tfrac12 d_j^2/S_{jj}$.
- **Asymptotic (long-delay) gain:**
  $$G = \frac{d^\top S^{-1} d}{\max_j d_j^2/S_{jj}}.$$
- **Fault in slice $j$ alone:** $G = S_{jj}(S^{-1})_{jj} = 1/(1-R_j^2)$, where $R_j^2$ is the squared multiple correlation of slice $j$ with the others.
- **Signals intuition:** the other slices act like the reference input of an adaptive noise canceller. Coupling helps only if the fault is *not* in the shared part of the noise.
  - A fault that moves the slices the way normal noise does gains nothing (gain exactly 1).
  - A fault that moves them in opposite directions gains the most.
- **2-slice example** (unit variance, correlation ρ):

  | shift (slice 1, slice 2) | ρ = 0.6 | ρ = 0.9 |
  |---|---|---|
  | (1, 0) | 1.56 | 5.26 |
  | (1, ρ) | 1.00 | 1.00 |
  | (1, 1) | 1.25 | 1.05 |
  | (1, −ρ) | 3.25 | 18.05 |

### 2.3 Autocorrelation, whitening and the zero-frequency (long-run) covariance

- **CUSUM theory assumes independent samples,** but queue latencies have lag-1 autocorrelation 0.6–0.8.
  - We whiten each stream with an AR(1) prediction-error filter (`StreamModel`), giving $z$.
  - That is the prewhitening step before a matched filter in coloured noise.
- **For a sustained (DC) fault the relevant noise covariance is the long-run covariance.** It is not the instantaneous one:
  $$\Sigma_{LR} = \sum_{\ell} \operatorname{Cov}(z_t, z_{t-\ell}) = 2\pi \times \text{cross-spectral density at frequency } 0,$$
  estimated with Bartlett weights over $L$ lags (`long_run_cov`). Equivalently, $\Sigma_{LR}$ with $L = n-1$ lags is $n$ times the covariance of $n$-window averages.
- **The zero-frequency gain** for a fault in slice $j$ alone is $1/(1-R^2_{j,\mathrm{DC}})$, where $R^2_{\mathrm{DC}}$ is the squared multiple *coherence* at frequency zero.
- **Joint whitening with a vector autoregression**, $x_t = \sum_\ell A_\ell x_{t-\ell} + e_t$ (`VARModel`):
  - a step shift $\mu$ appears in the innovations as $A(1)\mu$, with $A(1) = I - \sum_\ell A_\ell$;
  - since $\Sigma_{LR} = A(1)^{-1}\Sigma_e A(1)^{-\top}$, a CUSUM on the innovations has the same long-run rate as the $\Sigma_{LR}$ matched filter, with white increments;
  - per-slice whitening, however rich, cannot change a single stream's long-run rate $\mu_j^2/\Sigma_{LR,jj}$ (invariance under invertible filtering). Only the other slices can remove shared slow noise.

---

## 3. The simulator (`src/slicead/sim/`)

- **Model:**
  - slotted numpy model: 1 ms slots, KPIs per 100 ms window;
  - all runs are advanced together as arrays, and results are reproducible from a seed.
- **Resource sharing:** one server with work-conserving weighted water-filling (GPS per slot). Capacity follows an AR(1) fluctuation around nominal, standing for radio conditions.
- **Traffic:**
  - eMBB: on-off (Markov-modulated Poisson);
  - URLLC and mMTC: Poisson.
- **Losses:** tail-drop buffers (sized in ms of guaranteed rate) plus radio loss.
- **Delay:** exact fluid FIFO delay from the cumulative arrival and service curves (network-calculus style).
- **KPIs per slice:** throughput, latency, jitter, loss. Notebooks 06 onward also have a cell-level utilisation (like PRB utilisation).
- **Default scenario** (`configs/sim_default.yaml`):
  - 100 Mbit/s capacity (±10% AR(1) variation, 500 ms correlation time);
  - weights eMBB 0.5, URLLC 0.3, mMTC 0.2;
  - eMBB on-off 96/12 Mbit/s (200 ms on, 300 ms off); URLLC 2 Mbit/s; mMTC 8 Mbit/s;
  - offered load 56%.
  - All values are illustrative assumptions, not measurements.
- **Five settings** used in notebooks 11–13 (`configs/sim_*.yaml`):

  | setting | change from default | load |
  |---|---|---|
  | default | – | 56% |
  | high_load | bigger eMBB bursts (120/24 Mbit/s) | 72% |
  | new_mix | weights 0.3/0.3/0.4, 2× URLLC and mMTC traffic, slow eMBB bursts (1 s on, 1.5 s off) | 70% |
  | light_load | small eMBB bursts, 5% capacity variation | 38% |
  | control | constant capacity, Poisson eMBB: **no shared slow noise** | 38% |

- **Fault injection** (`Fault`, `configs/faults.yaml`): a fault multiplies one parameter (capacity, or a slice's traffic, radio loss or scheduling weight), as a step or a ramp. Runs carry labels. There are 7 scenarios:

  | scenario | what | expected type |
  |---|---|---|
  | capacity_drop | capacity ×0.7 | shared |
  | capacity_degradation | capacity ×0.7 as a 60 s ramp | shared |
  | embb_surge | eMBB traffic ×1.3 | shared |
  | urllc_surge | URLLC traffic ×5 | local |
  | mmtc_storm | mMTC traffic ×3 | local/competitive |
  | urllc_loss_burst | URLLC radio loss ×1000 | local (loss only) |
  | urllc_weight_misconfig | URLLC weight ×0.1 | competitive |

  **Magnitude sweep:** factor$^{s}$ for strength $s \in \{0.03, 0.1, 0.25, 0.5, 1\}$. For example, $s = 0.1$ of a ×5 surge is ×1.17.

---

## 4. Evaluation methodology (applies to every result)

- **Equal false-alarm rate.** Every detector is calibrated to the same ARL by simulation on separate fault-free runs (bisection on the threshold, with restarts). Multi-stream detectors use **equalised** per-stream or per-candidate thresholds, so that no single stream eats the false-alarm budget.
- **Delay:** conditional mean delay, with misses counted as the horizon; we also report the detection rate and the median delay.
- **No point-adjusted scoring**, which is known to inflate results (Kim et al., AAAI 2022).
- **Fixed seeds and common random numbers** across detectors; 32 or 64 runs per case.
- **Theory checks** on i.i.d. Gaussian data before trusting simulator results.
- **Pre-registration** where possible: notebook 13's predictions were written before the run.

---

## 5. Detectors implemented (`src/slicead/detectors/`)

| detector | what it is |
|---|---|
| `Shewhart` | alarm if any $\lvert z\rvert$ exceeds a threshold |
| `EWMA` | exponentially weighted moving average, λ = 0.2 |
| `CUSUM` | two-sided per-stream CUSUM, $k = 0.5$ |
| `CUSUMBank` | bank of coupled CUSUMs, one per candidate direction $d$, increment $z^\top S^{-1}d/\lVert d\rVert - k$ (Healy 1987; Pignatiello & Runger 1990). With $S = \Sigma$ it is the instantaneous bank; with $S = \Sigma_{LR}$ the zero-frequency bank; on VAR innovations with directions $A(1)d$ the VAR bank. `k` is configurable, and `CUSUMBank.matched(shift, Sigma)` gives a CUSUM matched to a known shift. |
| `candidate_directions` | 20 candidates for 3 streams: each slice alone, all together, each pair same-sign and opposite-sign, both signs |
| `MEWMA` | multivariate EWMA (Hotelling $T^2$ of the smoothed vector) |
| `Union` | runs several detectors on their own columns and alarms when any does; jointly calibrated |
| `StreamModel` | per-stream mean, sd, AR(1) fit; `standardize`, `whiten` |
| `VARModel` | per-stream AR(p) (`diagonal=True`) or joint VAR(p), order by BIC; `whiten`, `dc_gain` (= A(1)), `residual_directions` |
| `long_run_cov` | Bartlett estimate of $\Sigma_{LR}$ |

Evaluation helpers (`src/slicead/eval/metrics.py`): `arl_with_restarts`, `calibrate_threshold`, `calibrate_per_stream`, `calibrate_bank`, `calibrate_union`, `first_alarms`, `summarize_delays`.

---

## 6. Results, notebook by notebook

All notebooks are in `notebooks/`. Each exports its figures, tables and key numbers to `report/` (Section 10).

### 01 – CUSUM basics (i.i.d. Gaussian)
- Monte Carlo ARL and delay match Siegmund's approximation (for δ ≤ 1).
- The first-order log(ARL)/KL rule overestimates delay by about 1 − log KL; delay ≈ (log(KL·ARL) − 1)/KL.
- Autocorrelated (AR(1)) noise causes far more false alarms than theory predicts. So we whiten first, and calibrate by simulation.

### 02 – Coupled gain, Monte Carlo (2 streams)
- **Measured speed-ups are below the KL ratio.** At ARL 1000, ρ = 0.9:
  - shape (1,0): 3.75× (KL ratio 5.26);
  - shape (1,−ρ): 10.0× (KL ratio 18.05).
  - At ARL 10⁴ the same shapes give 4.15× and 11.9×. This was an early sign of the finite-ARL effect (notebooks 12–13).
- **Wrong shape is dangerous.** A coupled CUSUM tuned for (1,0) is 12.7× *slower* than single-slice CUSUM on (1,1), and 6.4× slower on (1,ρ). Hence a **bank over candidate shapes**.

### 03 – Simulator in normal operation
- **Coupling lives in latency, not throughput.** Cross-slice latency correlation is 0.62–0.89 raw and 0.41–0.72 after AR(1) whitening. Throughput is uncorrelated, because slices below their share deliver what arrives.
- Latency lag-1 autocorrelation is 0.6–0.8, and AR(1) whitening removes most of it.

### 04 – Fault signatures
- **Surges** show mainly in the slice's own throughput. The **loss burst** shows only in URLLC loss. **Capacity faults** move all latencies moderately.
- The **weight misconfiguration** sends URLLC latency up about 38 sd and mMTC latency *down* 2 sd.
- **First-order coupled latency gain** (instantaneous Σ): shared faults 1.03–1.07; one-slice faults 2.0–2.24.

### 05 – Detectors and magnitude sweep (11 KPIs, ARL 100 s)
- **I.i.d. thresholds are badly wrong** (ARL 26 windows instead of 1000), so calibration by simulation is essential.
- **Slice-local faults are easy** (URLLC surge ×1.17 in 0.4 s).
- **Shared capacity faults below about 8% are not detected** (chance level within 30 s is about 26%).
- **The URLLC loss burst is undetected;** it needs a count model.

### 06 – Utilisation KPI (negative result)
- Cell utilisation is load-driven and saturates during eMBB bursts.
- A full 30% capacity drop moves it by only 0.37 sd, so it does not help shared faults.

### 07 – Robust coupled detectors on the simulator (latency, ARL 100 s)
- **A shape-agnostic bank of 20 coupled CUSUMs beats per-slice CUSUM on small slice-local faults:**
  - URLLC surge, s = 0.25: 6.2×;
  - mMTC storm, s = 0.25: 11.6×;
  - weight misconfiguration, s = 0.1: 19× (0.4 s vs 7.5 s).
  - First-order theory predicted only about 2.1–2.2×.
- **The bank equals the oracle** on local faults, so not knowing the shape costs nothing there.
- **Shared faults:** 0.94–0.96×, no gain.
- **Against the full 11-KPI detector:** about 10× faster on the weight misconfiguration (0.4 s vs 4.1 s), a fault that changes no traffic.

### 08 – Combined detector (the recommended practical detector)
- **What it is:** per-KPI CUSUM on all 11 KPIs ∪ the coupled latency bank, jointly calibrated. Each part alone gets ARL 2000 windows, then one factor brings the union to 1000.
- **It matches the faster part on every fault** without knowing the fault type: median delay ratio to the better part 1.00, max 1.24.
- **Mean delay over the sweep:** 12.4 s, against 12.6 s for the per-KPI detector and 14.8 s for the bank.

### Demo (`demo/app.py`, Streamlit)
- **What it does:** simulates 60 s of three slices with a user-chosen fault and plays back KPIs and the statistics of three detectors (standard, coupled, combined).
- **Default case** (URLLC weight misconfiguration, s = 0.1): coupled alarm at about 0.6 s, standard at about 3.9 s.
- **Run it:** `python -m streamlit run demo/app.py --server.address localhost`.

### 09 – Why coupling beats Gaussian theory (mechanism)
- **Heavy-tail cancellation is refuted.** The common mode carries 70% of the variance but is only mildly non-Gaussian (excess kurtosis 0.66). Σ⁻¹ barely thins the URLLC/mMTC tails (kurtosis about 0.3 → 0.2).
- **The excess gain comes from the noise, not the signal.** On the simulator the regression residual is 4.0–12.6× faster. On Gaussian data with the same Σ and the same fault shift it is only 1.5–2.7×.
- **A shuffle test shows the 2× lower threshold comes from memory, not tails.** Shuffling time collapses it to 1.0.
- **The mechanism is low-frequency common-mode cancellation.** The relevant quantity is the coherence at zero frequency: gain 1/(1−R²_DC) ≈ 21–22, against the instantaneous 2.1–2.3.

### 10 – Zero-frequency coupled detectors (default setting, ARL 100 s)
- **Five detectors were compared:**
  - per-slice AR(1) and per-slice AR(p) (BIC p = 3);
  - the bank on Σ and the bank on Σ_LR;
  - the bank on joint VAR(3) innovations.
- **At s = 0.1, gain over per-slice AR(1) (mean delay):**
  - VAR bank **18.6–24.2×** (median-delay gains 16.5–21.3×);
  - Σ_LR bank 13–28×;
  - Σ bank 3.5–19×;
  - **richer per-slice whitening 0.8–1.0× (no help).**
- **Barely visible faults (s = 0.03):** the VAR bank finds 93–100% (weight misconfiguration in 1.6 s on average), against 41–75% for the Σ bank and about 30% per-slice.
- **Shared faults:** no gain; the VAR bank is somewhat worse on a medium capacity drop.
- **Robustness:** the Σ_LR bank is robust to the lag window L = 25–100.

### 11 – Five settings
- **Measured VAR-bank gain at the weakest strength with ≥ 80% per-slice detection:**
  - **control:** 0.94–1.21×, predicted 1.0–2.1, so correctly no gain;
  - **default / high load / new mix:** 14–33×, asymptotic rule 12.5–37 (measured/predicted 0.59–1.53); instantaneous theory 1.3–2.8, eight to fifteen times too low;
  - **light load:** 6.4–10.8×, against a predicted 15–39, which the rule *over-predicts* 2.4–4.8×.
- **Coherence builds up over different time-scales in each setting.** The Σ_LR estimate is ill-conditioned as R²_DC → 1: Bartlett and VAR-implied estimates differ by up to 3× in high load and new mix.

### 12 – Matched detectors: explaining the light-load miss
- **Design:** four *oracle* detectors that know the fault direction: best single slice and VAR innovations, each with fixed k = 0.5 or k matched to the fault size.
- **H1 (fixed reference k) is refuted.** Matching leaves light load over-predicted (measured/predicted 0.23–0.51).
- **Correction of notebook 11.** Matching moves gains in both directions elsewhere, so notebook 11's good agreement in three settings was partly the fixed-k handicap of the per-slice baseline.
- **H2 (short delays) is supported.**
  - Treat each matched detector as a Gaussian CUSUM with signal-to-noise $\mu = \sqrt{2\,\mathrm{KL}}$, using the long-run rates ($d_{j}^2/(2\Sigma_{LR,jj})$ for the slice; $\lVert\delta\rVert^2/2$ for the white VAR innovations).
  - Predict each delay with Siegmund at the actual ARL ($G_{\text{fin}}$ = ratio of predicted delays).
  - $G_{\text{fin}}$ is within ×2 of the measured gain in **30/30** cases (median 0.83), against 20/30 for the asymptotic ratio. In coupled settings without floor cases: 16/16 vs 9/16.

### 13 – ARL sweep (pre-registered test of H2)
- **Setup:** matched oracle detectors at ARL 100, 1000 and 10 000 windows, five settings, 23 usable cases.
- **Predictions written before the run, with outcomes:**
  - **P1 (gain rises with ARL in coupled settings): holds,** 17/17 cases; median ×2.64 from ARL 100 to 10 000 (predicted ×2.12).
  - **P2 (G_fin within ×2 at every ARL): holds at ARL 1000 and 10 000** (23/23). **At ARL 100, 21/23;** the two misses are high-load cases.
  - **P3 (control stays flat): holds,** 0.97–1.09 for single-slice faults and 1.4–2.0 for the two-slice weight misconfiguration (whose no-coupling value is about 2).
- **The asymptotic ratio becomes accurate only for long delays:** median measured/G_DC is 0.35, 0.65 and 0.92 at the three ARLs.
- **Open:** at ARL 10 000 the gain is 0.84 of the Bartlett limit but 0.46 of the VAR-implied one, and still rising, so which limit is right is unknown.

---

## 7. Corrections made along the way (do not repeat the old claims)

| old claim | status | replaced by |
|---|---|---|
| Coupling beats Gaussian theory because Σ⁻¹ cancels **heavy-tailed** common-mode noise (also suggested by an external Gemini literature review) | **Refuted** (nb 09) | Low-frequency common-mode cancellation: coherence at frequency zero |
| "The coupled gain is 1/(1−ρ²)" | Incomplete (nb 02) | Depends on the fault shape: $d^\top S^{-1}d / \max_j d_j^2/S_{jj}$ |
| "1/(1−R²_DC) predicts the realised gain" (nb 10–11) | **Refined** (nb 12–13) | It is the long-delay limit. At the operating ARL, convert the long-run rates into delays (finite-ARL $G_{\text{fin}}$). |
| nb 11's agreement in the default, high-load and new-mix settings | Partly coincidental (nb 12) | It was helped by the fixed-k per-slice baseline |

---

## 8. Prior art and novelty status

- **The detectors themselves are not new:**
  - directional/multivariate CUSUM (Healy 1987; Crosier 1988; Pignatiello & Runger 1990);
  - regression-adjusted charts (Hawkins 1991, 1993);
  - MEWMA (Lowry et al. 1992);
  - charting residuals of a VAR fit (e.g. Vanhatalo & Kulahci 2015; Lee, Park & Lee 2017);
  - long-run covariance in econometric sequential monitoring (e.g. Otto & Breitung, Backward CUSUM).
- **Related slicing work:**
  - Quan & Pathirana 2026 (shared resources as a common cause; attack *attribution*);
  - Mai et al. 2024 (inter-slice interference localisation).
  - Neither is about quickest detection.
- **What we have *not* found in targeted searches** (which cannot prove absence):
  - a slicing paper showing that slice KPIs are only moderately correlated instant by instant but about 95% coherent at low frequency;
  - one showing that the value of coupling for sustained faults follows the zero-frequency information rates at the operating ARL.
- **A systematic literature search and full-text reading are still needed before any novelty claim.**

---

## 9. Current claim (what a paper could say, with caveats)

> Under shared-resource slicing, slice KPIs are only moderately correlated at any instant (R² ≈ 0.3–0.8 across settings) but strongly coherent at low frequency (R²_DC ≈ 0.93–0.98), because slices share slow noise from traffic bursts and capacity drift. For sustained faults local to one slice, a CUSUM on jointly whitened (VAR) slices is 6–33× faster than per-slice CUSUM at the same false-alarm rate in every simulated setting with shared slow noise, and gives no gain where there is none. The gain is predicted by the zero-frequency information rates converted into delays at the operating ARL. It grows with the false-alarm interval towards the ratio of rates, as confirmed by a pre-registered ARL sweep. Instantaneous-correlation reasoning underestimates it by about an order of magnitude.

**Caveats:**
- simulation only, with illustrative parameters;
- the theory-matching experiments (nb 12–13) use oracle detectors that know the fault;
- practical detectors also pay for not knowing the fault (fixed k, candidate search);
- the estimate of the long-delay limit is uncertain (Bartlett vs VAR-implied);
- the finite-ARL prediction is weakest at very small ARLs;
- **shared faults (capacity drops) gain nothing from coupling** and remain hard;
- the URLLC loss burst needs a count model.

---

## 10. Repository, reproducibility and how to run

- **Repository:** this repository. The project's LaTeX write-up (`report/main.tex`) and its working notes are kept private and are not part of it.
- **Layout:**
  ```
  configs/            sim_default.yaml, sim_{high_load,new_mix,light_load,control}.yaml, faults.yaml
  src/slicead/sim/        simulator, configs, faults
  src/slicead/detectors/  online detectors (online.py), stream models (preprocess.py)
  src/slicead/eval/       calibration and delay metrics
  notebooks/          01-13 (Section 6)
  demo/               Streamlit app, prepare.py, calibration.json
  tests/              pytest suite (48 tests)
  report/             figures/ and tables/ written by the notebooks (created on first run; not committed)
  results/            notebook outputs (gitignored)
  ```
- **Setup:**
  ```bash
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
  .venv\Scripts\pip install -e .
  .venv\Scripts\python -m ipykernel install --sys-prefix --name topic7
  ```
- **Tests:** `.venv\Scripts\python -m pytest`
- **Run a notebook:** `.venv\Scripts\python -m nbconvert --to notebook --execute --inplace notebooks/NN_name.ipynb`, using the venv-local kernel `topic7`.
  - Do **not** use `python -m jupyter nbconvert`: on this laptop it picks up Anaconda's Python.
  - Notebook 13 caches each setting's rows in `results/13_arl_sweep/` and reuses them; delete them to recompute (same seeds, same numbers).
- **Exports:** each notebook writes `report/figures/NN_*.pdf`, `report/tables/NN_*.tex` and `NN_numbers.tex` macros, for the (private) LaTeX write-up. The notebooks themselves contain all results.
- **Pinned versions (Python 3.14):** numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, statsmodels 0.15.0, scikit-learn 1.9.1, matplotlib 3.11.2, streamlit 1.66.1.
- **Runtimes on the laptop:**
  - most notebooks take 3–10 min;
  - notebooks 11 and 12 take about 15–30 min;
  - notebook 13 takes about 25 min with three parallel jobs, or about 1 h sequentially.
  - Keep the laptop awake: a sleep killed one run.

---

## 11. What is not done yet (and next steps)

**Not done:**
- **ML baselines** (Isolation Forest, autoencoders) at the same ARL. sklearn is installed but unused in the detectors.
- **Real or public data**, e.g. the O-RAN `ue.csv` dataset used by Azkaei et al.
- **A count-based detector** for loss (the URLLC loss burst).
- **Shared-fault detection,** which needs information that separates capacity from load.
- **Shiryaev–Roberts, ARIMA/Kalman residuals and EVT-based tail monitoring** (only planned).

**Next steps, roughly in order of value for a paper:**
1. **Finite-ARL analysis for practical, non-oracle detectors** (bank with a fixed reference and candidate search).
2. **Put the VAR bank inside the combined detector and the demo.**
3. **ML baselines** at the same ARL.
4. **Systematic prior-art search** and full-text reading of the closest papers.
5. **The paper draft** (Dec–Feb).
6. **Practical items:** a GitHub remote; ask the professor whether simulation-only is acceptable; showcase slides.

---

## 12. References (with how far each was checked)

*Opened and checked:*
1. V. V. Veeravalli, T. Banerjee, *Quickest Change Detection* (tutorial), arXiv:1210.5552.
2. L. Xie, S. Zou, Y. Xie, V. V. Veeravalli, *Sequential (Quickest) Change Detection: Classical Results and New Directions*, 2021, arXiv:2104.04186.
3. S. Kim et al., *Towards a Rigorous Evaluation of Time-Series Anomaly Detection*, AAAI 2022, arXiv:2109.05257 (point-adjust inflates results).
4. A. Tartakovsky, A. Polunchenko, G. Sokolov, *Efficient Computer Network Anomaly Detection by Changepoint Detection Methods*, arXiv:1212.1829.
5. J. Paparrizos et al., *TSB-UAD*, PVLDB 2022 (code: github.com/TheDatumOrg/TSB-UAD).
6. Azkaei, Joshi, Exarchakos, *Machine Learning-Driven Anomaly Detection for 5G O-RAN Performance Metrics*, arXiv:2509.03290 (dataset `ue.csv` in o-ran-sc/ric-app-ad).
7. Rocha et al., *Optimal Resource Allocation with Delay Guarantees for Network Slicing in Disaggregated RAN*, arXiv:2305.17321.
8. Molina et al., *PUL-Inter-Slice Defender*, arXiv:2510.02236.

*Bibliographic details checked in citing sources or catalogue records; full text not read:*
- **Multivariate SPC:**
  - J. D. Healy, *A note on multivariate CUSUM procedures*, Technometrics 29(4) (1987) 409–412.
  - R. B. Crosier, *Multivariate generalizations of cumulative sum quality-control schemes*, Technometrics 30(3) (1988) 291–303.
  - J. J. Pignatiello, G. C. Runger, *Comparisons of multivariate CUSUM charts*, J. Quality Technology 22(3) (1990) 173–186.
  - C. A. Lowry, W. H. Woodall, C. W. Champ, S. E. Rigdon, *A multivariate exponentially weighted moving average control chart*, Technometrics 34(1) (1992) 46–53.
- **Regression adjustment:**
  - D. M. Hawkins, *Multivariate quality control based on regression-adjusted variables*, Technometrics 33(1) (1991) 61–75, doi:10.1080/00401706.1991.10484770.
  - D. M. Hawkins, *Regression adjustment for variables in multivariate quality control*, J. Quality Technology 25(3) (1993) 170–182, doi:10.1080/00224065.1993.11979451.
- **Autocorrelated and cross-correlated data:**
  - E. Vanhatalo, M. Kulahci, *The effect of autocorrelation on the Hotelling T² control chart*, QREI 31(8) (2015) 1779–1796, doi:10.1002/qre.1717.
  - K. Y. Lee, C. Park, M. L. Lee, *A multivariate CUSUM chart handling auto- and cross-correlated observations*, PHM Society Asia-Pacific 2017, doi:10.36001/phmap.2017.v1i1.1712 (use of VAR residuals not confirmed from the full text).
- **Long-run covariance and Gaussian processes:**
  - S. Otto, J. Breitung, *Backward CUSUM for testing and monitoring structural change…*, Econometric Theory, doi:10.1017/S0266466622000159; arXiv:2003.02682.
  - H. Keshavarz, C. Scott, X. Nguyen, *Optimal change point detection in Gaussian processes*, arXiv:1506.01338.
- **Slicing (abstracts only):**
  - V. S. Mai, R. La, T. Zhang, B. Hu, *Detection of performance interference among network slices in 5G/6G systems*, arXiv:2412.01584.
  - M. K. Quan, P. N. Pathirana, *Certified causal attribution for real-time attack forensics in 6G network slicing*, arXiv:2605.26679.

*Not confirmed (do not cite):* a "Biometrika heavy-tailed factor change detection 2025/2026" paper and "Qiu et al. 2018 Technometrics nonparametric multivariate CUSUM". Both appeared in an AI-generated literature review and could not be found.

---

## 13. Glossary

| term | meaning |
|---|---|
| eMBB / URLLC / mMTC | enhanced mobile broadband / ultra-reliable low-latency / massive machine-type communication slices |
| GPS | generalised processor sharing (weighted fair sharing of one server) |
| KPI | key performance indicator (throughput, latency, jitter, loss) |
| ARL | average run length to false alarm, in 100 ms windows (1000 = 100 s) |
| CUSUM | cumulative sum detector, $W_n = \max(0, W_{n-1} + \ell_n)$ |
| k | CUSUM reference value; matched value is half the shift |
| KL | Kullback–Leibler information per sample; delay ≈ log(ARL)/KL for long delays |
| whitening | removing a stream's autocorrelation with a prediction-error filter |
| Σ, Σ_LR | instantaneous and long-run (zero-frequency) noise covariance |
| R², R²_DC | squared multiple correlation / squared multiple coherence at frequency 0 |
| VAR | vector autoregression; joint whitening of all slices |
| G_DC | asymptotic coupling gain from Σ_LR (ratio of information rates) |
| G_fin | finite-ARL coupling gain (ratio of Siegmund-predicted delays at the operating ARL) |
| oracle detector | a detector told the fault's direction (and size, if matched); used to test theory |
| s | fault strength; the fault factor is raised to the power s |
