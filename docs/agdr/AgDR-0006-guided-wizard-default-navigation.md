---
id: AgDR-0006
timestamp: 2026-05-23T12:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Guided wizard as default Streamlit navigation

> In the context of **thesis demos and colleagues unfamiliar with eight Run tabs and five hardware-profile modes**, facing **navigation overload and Phase 4.5 vs Phase 5 confusion**, I decided **a six-page Guided demo (Intro → Steps 1–5) as the default `st.navigation` group, with the existing lab console under Advanced lab**, to achieve **an on-rails story without changing `hwa_cim` training semantics**, accepting **duplicate UX surface (wizard + Run page) until wizard paths fully subsume common demos**.

## Context

- AgDR-0002 established `st.navigation` + `st.Page` in `Home.py`.
- AgDR-0004/0005 require honest labeling of PEX, surrogate MC (Phase 4.5), and future Phase 5 CSV.
- User approved sequential slide-deck wizard (spec `docs/superpowers/specs/2026-05-23-guided-wizard-gui-design.md`).

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. Replace Run page entirely** | Single UI | Breaks power-user workflows |
| **B. Guided demo + Advanced lab groups** | On-rails default; CLI parity preserved | Two nav trees to maintain |
| **C. Single-page accordion wizard** | One file | Crowded; weak “slide” feel for demos |

## Decision

Chosen: **B**.

- `src/hwa_gui/pages/wizard/` — Intro, Baseline, Hardware, Noisy, HWA, Proof.
- `src/hwa_gui/wizard/` — state, copy, layout, actions, charts, runner (job wrappers only).
- Session gating via `wizard_step_max`; Quick vs Live demo modes.
- HWA profile mapping copied from Run tab 3a — no new training semantics.

## Consequences

- Default landing: **Intro** (old Home essay removed from nav).
- `GUI_RUN.md` documents Guided demo first.
- Phase 5 card disabled in Step 2; surrogate warnings match AgDR-0005.

## What not to claim

- Wizard does not implement foundry Phase 5 MC.
- Step 3 noisy eval does not switch noise tables per profile (profile affects Step 4).
