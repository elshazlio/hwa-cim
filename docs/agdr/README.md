# Agent Decision Records (AgDR)

This directory holds **Agent Decision Records** for the Thesis HW Codesign / **hwa-cim** project—structured notes on meaningful technical choices so reviewers and future agents understand *why*, not only *what* changed.

- **Format & philosophy:** [Agent Decision Records (Me2resh)](https://www.me2resh.com/blog/agent-decision-records), [spec / examples on GitHub](https://github.com/me2resh/agent-decision-record).
- **Cursor enforcement:** `.cursor/rules/agent-decision-records.mdc` (`alwaysApply: true`) reminds agents to add a record when decisions match the triggers there.

## Naming

`AgDR-{NNNN}-{short-kebab-title}.md`

Use four-digit zero-padded IDs. The next free number is **one higher than the largest existing `AgDR-NNNN`** (**0003** after AgDR-0002).

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
