---
id: AgDR-0002
timestamp: 2026-05-13T12:55:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Streamlit multipage navigation and human-readable sidebar labels

> In the context of **the SRAM HWA Lab Streamlit GUI** using a `pages/` directory next to `Home.py`, facing **sidebar labels tied to filenames (`1_Run.py`) and no native `[pages]` table in `config.toml`**, I decided **`st.navigation` + `st.Page(..., title=..., icon=...)` in `Home.py`, explicit `[client] showSidebarNavigation = true`, a two-line comment in `.streamlit/config.toml` pointing authors to `Home.py`, per-page `st.set_page_config` for browser tab titles, and `streamlit>=1.52` in optional `[gui]` deps**, to achieve **sidebar entries that read “Home”, “Run”, “Results”, “Compare”, “Charts”, “Noise profile” without duplicating manual `st.page_link` blocks**, accepting **entrypoint-only routing (no `pages/` auto-discovery alongside navigation), a minimum Streamlit bump from 1.33 → 1.52, and documentation that `[pages]` is not a supported Streamlit config section**.

## Context

- Thesis students and demos need **obvious navigation**; numeric filenames alone are easy to misread.
- A request was made to set labels via **“one-line Streamlit `[pages]` title in config”**. Upstream Streamlit’s `config.toml` documents **`[client]`**, **`[theme]`**, **`[server]`**, etc., but **does not define `[pages]`** for renaming files discovered under `pages/`.
- The prior GUI duplicated navigation via **`st.page_link`** in `render_pipeline_sidebar`, which cluttered the sidebar once a proper nav exists.
- **Project rule:** non-obvious choices affecting the **CLI / GUI boundary** and **dependency bounds** belong in an AgDR.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. `[pages]` in `config.toml` only** | Matches mistaken mental model if it existed | **Not supported** by Streamlit; would be dead config or risk confusion. |
| **B. Rename `pages/*.py` only** (e.g. `1_-_Run.py`) | No code router; uses built-in filename→label rules | Still awkward for “Noise profile” vs file naming; less control over icons/order than `st.Page`. |
| **C. `st.navigation` + `st.Page` in `Home.py`** | Official v2 multipage API; explicit **title** and **icon**; clean URLs (`/Run`, …) | **Disables `pages/` auto-discovery** in favor of declared pages; requires **Streamlit ≥ ~1.52**; entry `Home.py` must stay the router. |
| **D. Third-party `st_pages` / `.streamlit/pages.toml`** | TOML-driven labels without router code | Extra dependency and maintenance; overkill for six pages. |

## Decision

Chosen: **C**, plus:

- **`.streamlit/config.toml`**: two-line comment that sidebar titles live in `Home.py` (since **`[pages]` is not a real Streamlit section**); **`[client] showSidebarNavigation = true`** for explicit default-on nav.
- **`pyproject.toml`**: optional GUI extra **`streamlit>=1.52`** (was `>=1.33`).
- **Each page script**: **`st.set_page_config` as the first Streamlit call** so browser tabs read e.g. `Run · SRAM HWA Lab`.
- **`render_pipeline_sidebar`**: remove duplicate **`st.page_link`** list; keep progress ✓/○ and copy; point users to the **built-in nav** at the top of the sidebar.

Rejected: **A** (non-functional as specified). Rejected for now: **D** (dependency). **B** kept as fallback mental model only (filename rules remain documented in Streamlit docs).

## Consequences

- Installers must use **`pip install -e ".[gui]"`**, which now pulls **Streamlit ≥ 1.52** (needed for `st.navigation` / `st.Page`).
- New pages are added in **two places**: the script under `pages/` **and** a **`st.Page(...)` line in `Home.py`**.
- **AgDR discipline:** GUI navigation and Streamlit floor change are recorded here so future agents do not “fix” this by inventing `[pages]` in `config.toml` or downgrading Streamlit without review.
