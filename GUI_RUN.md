# SRAM HWA Lab — Streamlit dashboard

**Repo:** [`hwa-cim`](https://github.com/elshazlio/hwa-cim) · **Package:** `hwa-cim`

The **`hwa-dashboard`** command opens the **SRAM HWA Lab**: a Streamlit UI for hybrid (analog + digital) SRAM hardware-aware training—the same phases as the CLI.

## Start here: Guided demo (default)

The sidebar opens **Guided demo** — a six-step walkthrough for colleagues and thesis demos:

| Step | What you do |
| ---- | ----------- |
| **Intro** | 30-second CiM + HWA story → **Start demo** |
| **1 · Baseline** | Load or train Phase 1 (`results/run_baseline/`) |
| **2 · Hardware** | Pick profile: Synthetic, Maestro PEX, **Phase 4.5 Surrogate MC**, corners; Phase 5 shown as **future work** |
| **3 · Noise crash** | Noisy eval on baseline (γ=0.02) |
| **4 · HWA recovery** | Train/load HWA with the Step 2 profile |
| **5 · Thesis proof** | Three-bar chart + safe/unsafe claims |

**Quick demo** loads existing `results/` artifacts. **Live demo** runs training jobs (one at a time, same as Advanced **Run**).

**Advanced lab** (second sidebar group) keeps the full console: **Run**, **Results**, **Compare**, **Charts**, **Hardware profiles**.

**Thesis slide figures:** In **Charts → Thesis slide figures (deck)** or wizard Steps 2/5, click **Generate thesis slide figures**, or run `hwa-plot-thesis-slides`. Outputs land in `results/plots/04_thesis_slides/`.

Navigation is defined in **`src/hwa_gui/Home.py`** (`st.navigation` groups). See **`docs/agdr/AgDR-0006-guided-wizard-default-navigation.md`**. Optional `[gui]` installs **Streamlit ≥ 1.52**.

**Use your real clone path** (folder that contains `pyproject.toml`).  
If `cd` fails, do not run `pip install` from `~` — you may see `file:///Users/... does not appear to be a Python project`.

**Do not paste HTML-style arrows (`<--`) on the same line as a command** unless they are inside a `#` shell comment. If `#` is missing, zsh treats `<` as redirection and you get errors like `no such file or directory: --`.

## First time (create venv and install)

Recommended (tests + dashboard):

```bash
cd "$HOME/Documents/My Projects/hwa-cim"
ls pyproject.toml
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"
```

Change the `cd` line if your clone lives elsewhere. If `ls pyproject.toml` fails, fix `cd` before running `pip`.

Dashboard only (no pytest extra — activate the venv first, then):

```bash
pip install -e ".[gui]"
```

On **Windows** (PowerShell):

```powershell
cd "C:\Users\YOURNAME\Documents\hwa-cim"
Get-Item pyproject.toml
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

## Every time (activate venv and launch)

**macOS / Linux:**

```bash
cd "$HOME/Documents/My Projects/hwa-cim"
source .venv/bin/activate
hwa-dashboard
```

**Windows (PowerShell):**

```powershell
cd "C:\Users\YOURNAME\Documents\hwa-cim"
.\.venv\Scripts\Activate.ps1
hwa-dashboard
```

Your default browser opens automatically when the server is ready (usually `http://localhost:8501`). The URL is also printed in the terminal.

**Equivalent** (from repo root, venv active, same working directory behavior as `hwa-dashboard`):

```bash
python -m streamlit run src/hwa_gui/Home.py
```

`hwa-dashboard` changes the process cwd to the repo root so `data/` and `results/` match the README examples.

---

## What each Advanced lab page does

| Page | Role |
| ---- | ---- |
| **Run** | Tabs in thesis order: baseline → noisy eval / Γ sweep → HWA train / HWA sweep → distill → thesis PNG → parasitic PNG. Each tab has an expander (“what this does / what you need”). |
| **Results** | Pick a `metrics.json`; list checkpoints, CSVs, JSON, PNGs under `results/`. |
| **Compare** | Multi-select `metrics.json` files → one table (good for Phase 1 vs Phase 3). |
| **Charts** | Interactive Plotly (parasitic, gamma sweep CSV, HWA sweep CSV, thesis bars)—same data as some **Run** figure tabs. |
| **Hardware profiles** | **Synthetic** (default), **Maestro PEX** (deterministic `/OA_Charge` cal — *not* MC σ), **PEX corners** (proxy only), **Phase 4.5 Surrogate MC** (wide VIVA parser + plots — *not* foundry MC), **Monte Carlo CSV** (true Phase 5). Run **3a** uses a **hardware profile mode** banner so modes are not confused. |

**Do not treat Maestro PEX or Phase 4.5 surrogate as Phase 5:** Maestro PEX writes `calibration_pex.yaml` and waveform PNGs. Phase 4.5 runs `hwa-surrogate-mc` and `hwa-plot-surrogate-mc` for thesis artifacts; HWA still trains with **synthetic** γ unless you pick **True Monte Carlo CSV**. The thesis three-bar chart (**Run → Fig · Thesis bars**) is **FP32 / INT4+noise / HWA**, not “PEX drop → HWA recovery.”

The **sidebar** on every page has the **built-in page navigation** at the top (labels from `Home.py`), then a short lab description, a **✓/○ checklist** for *default* artifact paths (`results/run_baseline/best.pt`, `results/run_baseline/noisy_eval.json`, `results/run_hwa/best.pt`), and a **suggested next step**. If you use custom output directories, rely on **Results**; the checklist is only a hint for the default layout.

**Adding a page:** create a script under `src/hwa_gui/pages/`, then register it with **`st.Page(...)`** in **`Home.py`** (`_navigation_pages()`), or it will not appear in the nav.

---

## Run page behavior

- **One job at a time** on the Run page. Wait for the current job to finish before starting another.
- If the output directory already has files, you must **confirm overwrite** via the checkbox before the run starts.
- **Logs** stream at the bottom of the Run page (`results/.dashboard_last.log` is updated for the live tail).
- **Partial runs:** each tab writes to disk. You can stop after any step and later open the tab that matches what you have on disk (e.g. skip to HWA if `best.pt` already exists from a previous CLI run).

**Run tab labels (CLI parity):** `1 · Baseline`, `2a · Noisy eval`, `2b · Gamma sweep`, `3a · HWA train`, `3b · HWA sweep`, `4 · Distill`, `Fig · Thesis bars`, `Fig · Parasitic`.

The GUI’s **Noisy eval** tab defaults the optional JSON path to `results/run_baseline/noisy_eval.json` so it lines up with the README quick start and the thesis plot; the CLI still accepts `--out` as you prefer.

---

## Metrics keys (Phase 1)

Phase 1 **`metrics.json`** includes **`int4_ptq_test_accuracy_ideal`** and **`int4_ptq_test_accuracy_hardware`**. Older runs may only have **`int4_ptq_test_accuracy`**. See **`docs/agdr/AgDR-0001-hardware-aware-mac-calibration.md`**.

---

## Already installed?

If `.venv` exists and dependencies are installed:

```bash
cd "$HOME/Documents/My Projects/hwa-cim"
source .venv/bin/activate
hwa-dashboard
```
