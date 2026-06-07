# Agent Decision Records (AgDR)

This directory holds **Agent Decision Records** for **hwa-cim** ([`elshazlio/hwa-cim`](https://github.com/elshazlio/hwa-cim), Python package **`hwa-cim`**)—structured notes on meaningful technical choices so reviewers and future agents understand *why*, not only *what* changed.

- **Format & philosophy:** [Agent Decision Records (Me2resh)](https://www.me2resh.com/blog/agent-decision-records), [spec / examples on GitHub](https://github.com/me2resh/agent-decision-record).
- **Cursor enforcement:** `.cursor/rules/agent-decision-records.mdc` (`alwaysApply: true`) reminds agents to add a record when decisions match the triggers there.

## Naming

`AgDR-{NNNN}-{short-kebab-title}.md`

Use four-digit zero-padded IDs. The next free number is **one higher than the largest existing `AgDR-NNNN`** (**0007** after AgDR-0006).

## Workflow

1. Copy [TEMPLATE.md](TEMPLATE.md).
2. Fill frontmatter and all sections; write the Y-statement first if it helps clarify thinking.
3. Set `status: proposed` while iterating; set `executed` when the change is merged or accepted.
4. If a decision is replaced later, set the old file to `superseded` and add `supersedes:` in the new file.

## Index

| ID | Title | Status |
|----|--------|--------|
| [AgDR-0001](AgDR-0001-hardware-aware-mac-calibration.md) | Hardware-aware MAC calibration in software | executed |
| [AgDR-0002](AgDR-0002-streamlit-navigation-and-sidebar-labels.md) | Streamlit navigation and sidebar labels | executed |
| [AgDR-0003](AgDR-0003-yaml-calibration-and-csv-noise.md) | YAML MAC calibration and richer CSV noise | executed |
| [AgDR-0004](AgDR-0004-maestro-pex-calibration-path.md) | Maestro PEX calibration path separate from MC CSV | executed |
| [AgDR-0005](AgDR-0005-surrogate-monte-carlo-profile-mode.md) | Phase 4.5 surrogate MC (user-defined Gaussian parametric variation) | executed |
| [AgDR-0006](AgDR-0006-guided-wizard-default-navigation.md) | Guided wizard as default Streamlit navigation | executed |
| [AgDR-0007](AgDR-0007-cadence-informed-surrogate-stress.md) | Cadence-informed surrogate stress training mode | executed |
