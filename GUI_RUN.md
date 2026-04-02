# Run the HWA-CiM lab dashboard

From the **repository root** (the folder that contains `pyproject.toml`):

## First time (create venv and install)

```bash
cd "/Users/elshazlio/Documents/My Projects/Thesis HW Codesign"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui]"
```

Adjust the `cd` path if your project lives elsewhere. On **Windows** (PowerShell):

```powershell
cd "C:\path\to\Thesis HW Codesign"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[gui]"
```

## Every time (activate venv and launch)

**macOS / Linux:**

```bash
cd "/Users/elshazlio/Documents/My Projects/Thesis HW Codesign"
source .venv/bin/activate
hwa-dashboard
```

**Windows (PowerShell):**

```powershell
cd "C:\path\to\Thesis HW Codesign"
.\.venv\Scripts\Activate.ps1
hwa-dashboard
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

**Equivalent without the `hwa-dashboard` script** (from repo root, venv active):

```bash
python -m streamlit run src/hwa_gui/Home.py
```

## Already installed?

If `.venv` exists and dependencies are installed, only this is required:

```bash
cd "/Users/elshazlio/Documents/My Projects/Thesis HW Codesign"
source .venv/bin/activate
hwa-dashboard
```

