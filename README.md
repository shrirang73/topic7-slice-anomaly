# Quickest detection of QoS violations in 5G network slices

5G Hackathon, Topic 7 (Communication Systems, BITS Pilani Goa). Team: Shrirang Bhale, Palash, Naveen.

5G network slices (eMBB, URLLC, mMTC) share one physical resource pool, so a fault in one slice shows up in the KPIs of the others. We simulate slices as queues sharing a server, inject labelled faults, and compare detectors on **detection delay at an equal false-alarm rate** (ARL), with every threshold calibrated by simulation.

**Detectors implemented:**
- threshold (Shewhart), EWMA, per-stream CUSUM;
- coupled detectors: a shape-agnostic bank of multivariate CUSUMs, MEWMA, banks on long-run covariance and on VAR-whitened streams;
- a combined detector.

**Main finding so far.**
- Slices share slow, low-frequency noise, from traffic bursts and capacity drift.
- A coupled detector on jointly whitened latencies is **6–33× faster** than per-slice CUSUM on small slice-local faults, in every simulated setting with shared slow noise. It gives no gain where there is none.
- The size of the gain is predicted by the zero-frequency information rates converted into delays at the operating false-alarm rate.
- Everything here is simulation with illustrative parameters.

Everything runs on an ordinary laptop: no hardware, CPU only. A detailed, self-contained description of the project, its maths and every result is in [docs/project_reference.md](docs/project_reference.md).

| Notebook | What it shows |
|---|---|
| `notebooks/01_cusum_basics.ipynb` | CUSUM from scratch. The Monte Carlo ARL and delay match Siegmund's approximation; the log(ARL)/KL rule overestimates delay; correlated noise causes many more false alarms. |
| `notebooks/02_coupled_gain.ipynb` | A coupled 2-slice CUSUM is faster only when the fault breaks the normal correlation, and by less than the KL ratio. Tuned to the wrong fault shape, it can be 10× slower than a single-slice CUSUM. |
| `notebooks/03_slice_simulator.ipynb` | The 3-slice shared-server simulator (`src/slicead/sim/`) in normal operation: KPI traces, coupling across slices, memory and distribution shape. |
| `notebooks/04_fault_signatures.ipynb` | Seven labelled fault scenarios (`configs/faults.yaml`). Each fault's signature across all KPIs, and the latency coupling gain: about 1 for shared faults, about 2 for faults concentrated in one slice. |
| `notebooks/05_detectors.ipynb` | Shewhart, EWMA and per-stream CUSUM detectors (`src/slicead/detectors/`), calibrated by simulation to a common ARL, over a fault-magnitude sweep. Slice-local faults are caught within about a second; shared capacity faults below about 8% are not. |
| `notebooks/06_utilisation_kpi.ipynb` | Adds a cell-level resource-utilisation KPI (like PRB utilisation). Negative result: it is dominated by load and saturates during bursts, so it does not make shared capacity faults detectable. |
| `notebooks/07_coupled_detectors.ipynb` | A shape-agnostic bank of coupled latency CUSUMs is 6–19× faster than per-slice CUSUM on small slice-local faults (far above the first-order 2×), and about 10× faster than the best all-KPI detector on a scheduling-weight misconfiguration. No gain on shared faults. |
| `notebooks/08_combined_detector.ipynb` | **Recommended detector.** It combines the per-KPI CUSUM and the coupled latency bank, jointly calibrated to the same false-alarm rate, and matches the faster of the two on every fault (median delay ratio 1.00) without knowing the fault type. |
| `notebooks/09_tail_cancellation.ipynb` | Why coupling beats Gaussian theory. Not heavy tails (refuted): the slices share slow, low-frequency noise, and the gain is governed by the cross-slice coherence at frequency zero. |
| `notebooks/10_long_run_coupling.ipynb` | Detectors built on that mechanism. A bank on jointly VAR-whitened latencies is about 17–24× faster than per-slice CUSUM on small slice-local faults, matching the zero-frequency prediction 1/(1−R²_DC) ≈ 20. Richer per-slice whitening gives nothing. |
| `notebooks/11_second_load.ipynb` | The zero-frequency rule in five settings (`configs/sim_*.yaml`). It predicts no gain where slices share no slow noise (correct), and the 14–33× gain to within a factor of 1.7 in three coupled settings. It over-predicts 2.4–4.8× at light load, which is still unexplained. Instantaneous correlation underestimates the gain about 10× wherever slices share slow noise. |
| `notebooks/12_matched_reference.ipynb` | Why the rule over-predicted at light load. Matching each CUSUM to the fault's size does not explain it. What does: the rule is a long-delay limit. Converting the same zero-frequency information rates into delays at the actual ARL (Siegmund) predicts matched detectors' coupling gain to within a factor of 2 in all 30 cases across five settings. |
| `notebooks/13_arl_sweep.ipynb` | Pre-registered test of that explanation at ARL 100, 1000 and 10 000. The coupling gain rises with the ARL in 17 of 17 coupled cases (median ×2.6). The finite-ARL prediction holds at ARL 1000 and 10 000 in 23 of 23 cases, and at ARL 100 in 21 of 23. The control stays flat. Saves per-setting results in `results/13_arl_sweep/` and reuses them on a re-run. |

## Demo

An interactive Streamlit app (`demo/app.py`) simulates 60 s of three slices, injects a fault you choose, and shows three detectors raising their alarms live: the standard per-KPI detector, the coupled latency detector, and the combined detector that unites them. Each alarm shows its delay and the symptom that tripped it. It runs locally; no internet or hardware is needed.

```bash
python demo/prepare.py
```

Run that once (about 3 minutes) to calibrate both detectors into `demo/calibration.json`. The file is committed, so this is only needed after changing the simulator or detectors. Then start the app:

```bash
python -m streamlit run demo/app.py --server.address localhost
```

and open http://localhost:8501. `--server.address localhost` keeps the app private to your laptop. The default scenario, a small URLLC scheduling-weight misconfiguration, shows the project's main result: the coupled detector alarms within about 0.6 s, the standard one after about 4 s.

## Setup

Requires Python 3.12+ (developed on 3.14).

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --sys-prefix --name topic7 --display-name "Python 3 (topic7 .venv)"
```

Linux / macOS / Git Bash:
```bash
python -m venv .venv
source .venv/bin/activate   # Git Bash on Windows: source .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --sys-prefix --name topic7 --display-name "Python 3 (topic7 .venv)"
```

`pip install -e .` makes the project package `slicead` (in `src/`) importable from the notebooks. The last line registers the venv as the Jupyter kernel `topic7`, which the notebooks use.

Run the tests with `python -m pytest`. Re-run a notebook with the following command. Use `python -m nbconvert`, not `python -m jupyter nbconvert`: the latter can pick up a different Jupyter installation (e.g. Anaconda) that is earlier on PATH.

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/03_slice_simulator.ipynb
```

## Layout
```
configs/        simulator scenarios (sim_*.yaml) and fault scenarios (faults.yaml)
src/slicead/    the package: sim/ (simulator, faults), detectors/ (online detectors, stream models), eval/ (calibration, delays)
notebooks/      01-13, one experiment each (table above)
demo/           Streamlit app, calibration script and calibration.json
tests/          pytest suite
docs/           project_reference.md
results/        generated outputs (not committed)
report/         figures/ and tables/ written by the notebooks (not committed; created when a notebook runs)
```

## Conventions
- Fixed random seeds; experiments read config files and save their outputs to `results/`.
- Every detector is compared at the same simulated false-alarm rate; no point-adjusted scoring.
