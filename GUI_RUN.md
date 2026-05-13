# HWA-CiM Lab — Streamlit dashboard

The **`hwa-dashboard`** command opens the **HWA-CiM Lab**: a Streamlit UI around the same training, evaluation, and plotting entrypoints as the CLI. Use it to **launch jobs**, **tail logs**, **browse `results/`**, **compare `metrics.json`**, preview **Plotly** charts, and **validate Phase 5 noise CSVs**—especially useful for demos and sanity checks. Nothing replaces the CLI for scripting; the GUI is optional.

**Multipage layout:** sidebar labels (**Home**, **Run**, **Results**, …) come from **`st.navigation` / `st.Page`** in `src/hwa_gui/Home.py`, not from a `[pages]` table in `config.toml` (Streamlit does not support that). See **`docs/agdr/AgDR-0002-streamlit-navigation-and-sidebar-labels.md`**. Optional `[gui]` installs **Streamlit ≥ 1.52**.

**Use your real clone path** (folder that contains `pyproject.toml`).  
If `cd` fails, do not run `pip install` from `~` — you may see `file:///Users/... does not appear to be a Python project`.

**Do not paste HTML-style arrows (`<--`) on the same line as a command** unless they are inside a `#` shell comment. If `#` is missing, zsh treats `<` as redirection and you get errors like `no such file or directory: --`.

## First time (create venv and install)

Recommended (tests + dashboard):

```bash
cd "$HOME/Documents/My Projects/Thesis HW Codesign"
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
cd "C:\Users\YOURNAME\Documents\Thesis HW Codesign"
Get-Item pyproject.toml
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

## Every time (activate venv and launch)

**macOS / Linux:**

```bash
cd "$HOME/Documents/My Projects/Thesis HW Codesign"
source .venv/bin/activate
hwa-dashboard
```

**Windows (PowerShell):**

```powershell
cd "C:\Users\YOURNAME\Documents\Thesis HW Codesign"
.\.venv\Scripts\Activate.ps1
hwa-dashboard
```

Streamlit prints a local URL (usually `http://localhost:8501`).

**Equivalent** (from repo root, venv active, same working directory behavior as `hwa-dashboard`):

```bash
python -m streamlit run src/hwa_gui/Home.py
```

`hwa-dashboard` changes the process cwd to the repo root so `data/` and `results/` match the README examples.

---

## What each page does

| Page | Role |
| ---- | ---- |
| **Home** | Short orientation, “where to start”, phase table, link to **AgDR-0001** for INT4 metric keys, health strip. |
| **Run** | Tabs in thesis order: baseline → noisy eval / Γ sweep → HWA train / HWA sweep → distill → thesis PNG → parasitic PNG. Each tab has an expander (“what this does / what you need”). |
| **Results** | Pick a `metrics.json`; list checkpoints, CSVs, JSON, PNGs under `results/`. |
| **Compare** | Multi-select `metrics.json` files → one table (good for Phase 1 vs Phase 3). |
| **Charts** | Interactive Plotly (parasitic, gamma sweep CSV, HWA sweep CSV, thesis bars)—same data as some **Run** figure tabs. |
| **Noise profile** | Validate and preview **Phase 5** Monte Carlo / hardware CSV before **Run → HWA train** with noise mode **csv**. |

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
cd "$HOME/Documents/My Projects/Thesis HW Codesign"
source .venv/bin/activate
hwa-dashboard
```
