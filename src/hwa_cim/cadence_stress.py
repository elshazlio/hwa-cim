"""
Cadence-informed surrogate stress from Phase 4.5 summaries (AgDR-0007).

Maps normalized /OA_Charge spread to relative output noise for training/eval.
Not Phase 5 per-code Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PHASE_LABEL = "Phase 4.5"
PROFILE_KIND = "cadence_informed_surrogate_stress"
PROFILE_DISPLAY_NAME = "Cadence-informed surrogate stress"
DEFAULT_WARNING = (
    "Uses normalized /OA_Charge surrogate statistics; "
    "not foundry Monte Carlo or final Phase 5"
)

_PHASE5_REQUIRED = frozenset({"input_code", "ideal_output", "mean_output", "sigma"})
_PHASE45_SUMMARY = frozenset({"mean_output", "sigma_output"})


@dataclass(frozen=True)
class CadenceStressProfile:
    """Phase 4.5 summary row → normalized output stress."""

    source_summary: Path
    marker: str
    variable_group: str
    mean_output: float
    sigma_output: float
    spread_output: float
    surrogate_sigma_rel: float
    profile_kind: str = PROFILE_KIND
    phase_label: str = PHASE_LABEL
    profile_warning: str = DEFAULT_WARNING
    stress_scale: float = 1.0

    def metrics_fields(self) -> dict[str, object]:
        return {
            "surrogate_summary": str(self.source_summary.resolve()),
            "surrogate_marker": self.marker,
            "surrogate_variable_group": self.variable_group,
            "mean_output": self.mean_output,
            "sigma_output": self.sigma_output,
            "spread_output": self.spread_output,
            "surrogate_sigma_rel": self.surrogate_sigma_rel,
            "stress_scale": self.stress_scale,
            "phase_label": self.phase_label,
            "profile_kind": self.profile_kind,
            "profile_is_foundry_certified": False,
            "profile_warning": self.profile_warning,
        }


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower().strip(): c for c in df.columns}


def _reject_phase5_schema(cols: dict[str, str], path: Path) -> None:
    """Reject per-code Phase 5 Monte Carlo CSVs."""
    if _PHASE5_REQUIRED.issubset(cols):
        raise ValueError(
            f"{path} looks like a Phase 5 per-code Monte Carlo profile "
            f"(columns {sorted(_PHASE5_REQUIRED)}). "
            "Use --noise-mode csv with NoiseProfileCSV, not cadence_stress."
        )
    if "input_code" in cols and "sigma" in cols and "sigma_output" not in cols:
        raise ValueError(
            f"{path} has input_code + sigma but not Phase 4.5 sigma_output — "
            "refusing to treat as surrogate stress summary."
        )


def load_cadence_stress_profile(
    path: Path,
    *,
    stress_scale: float = 1.0,
    marker: str | None = None,
) -> CadenceStressProfile:
    """
    Load Phase 4.5 ``surrogate_mc_summary.csv`` and compute relative output stress.

    ``surrogate_sigma_rel = (sigma_output / |mean_output|) * stress_scale``
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty summary CSV: {path}")

    cols = _normalize_columns(df)
    _reject_phase5_schema(cols, path)

    if not _PHASE45_SUMMARY.issubset(cols):
        raise ValueError(
            f"{path} missing Phase 4.5 summary columns "
            f"{sorted(_PHASE45_SUMMARY)}; got {list(df.columns)}"
        )

    work = df
    if marker is not None and "marker" in cols:
        mcol = cols["marker"]
        subset = work[work[mcol].astype(str) == marker]
        if subset.empty:
            raise ValueError(f"No row with marker={marker!r} in {path}")
        work = subset

    row = work.iloc[0]
    mean_out = float(row[cols["mean_output"]])
    sigma_out = float(row[cols["sigma_output"]])
    if abs(mean_out) < 1e-12:
        raise ValueError(f"|mean_output| too small for relative sigma in {path}")

    spread = float(row[cols["spread_output"]]) if "spread_output" in cols else float("nan")
    sigma_rel = (sigma_out / abs(mean_out)) * float(stress_scale)

    return CadenceStressProfile(
        source_summary=path,
        marker=str(row[cols["marker"]]) if "marker" in cols else "default",
        variable_group=str(row[cols["variable_group"]]) if "variable_group" in cols else "",
        mean_output=mean_out,
        sigma_output=sigma_out,
        spread_output=spread,
        surrogate_sigma_rel=sigma_rel,
        stress_scale=float(stress_scale),
        profile_warning=str(row[cols["profile_warning"]])
        if "profile_warning" in cols
        else DEFAULT_WARNING,
    )
