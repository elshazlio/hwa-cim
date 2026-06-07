"""Hardware profiles: synthetic, Maestro PEX, corners, Monte Carlo CSV."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hardware profiles · SRAM HWA Lab",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_cim.maestro_pex import (
    DEFAULT_MANIFEST,
    DEFAULT_PEX_CALIBRATION,
    MaestroRunSpec,
    process_corner_runs,
    process_run_pair,
    run_maestro_pex,
)
from hwa_cim.noise import NoiseProfileCSV

from hwa_gui.components import (
    apply_page_style,
    render_hardware_profile_banner,
    render_pipeline_sidebar,
    root,
)
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()
render_pipeline_sidebar(current="Hardware profiles")

st.title("Hardware profiles")
st.caption(
    "Choose how noise and calibration enter HWA training. "
    "Only **Monte Carlo CSV** is the thesis-grade statistical Phase 5 path."
)

tab_syn, tab_pex, tab_corner, tab_sur, tab_cadence, tab_mc = st.tabs(
    [
        "Synthetic",
        "Maestro PEX",
        "PEX corners",
        "Phase 4.5 Surrogate MC",
        "Cadence-informed stress",
        "Monte Carlo CSV",
    ]
)

with tab_syn:
    render_hardware_profile_banner("synthetic")
    st.markdown(
        """
**Synthetic AFM noise** — default simulation path:

- Training uses `gamma` · max|W| Gaussian weight noise and optional ADC quantization.
- Optional schematic calibration: `config/calibration.yaml` (gain/offset ladder knobs).
- Does **not** read Cadence waveform or Monte Carlo exports.

Use **Run → 3a HWA train** with hardware profile **Synthetic AFM Noise**.
"""
    )

with tab_pex:
    render_hardware_profile_banner("maestro_pex")
    st.warning(
        "**Read_Out_*** is ignored for this testbench because read mode and the sense amp are not active."
    )
    c1, c2 = st.columns(2)
    nopex = c1.text_input("No-PEX CSV", value="stuff_from_cadence/nopex1.csv", key="mp_nopex")
    pex = c2.text_input("PEX CSV", value="stuff_from_cadence/pex1.csv", key="mp_pex")
    c3, c4, c5 = st.columns(3)
    signal = c3.text_input("Signal", value="/OA_Charge", disabled=True, key="mp_sig")
    sample_ns = c4.number_input("Sample time (ns)", 0.0, 1e6, 200.0, key="mp_t")
    out_dir = Path(c5.text_input("Output dir", value="results/maestro_pex", key="mp_out"))
    cal_out = st.text_input(
        "Calibration YAML (optional)",
        value=str(DEFAULT_PEX_CALIBRATION.relative_to(root())),
        key="mp_cal",
    )
    use_manifest = st.checkbox("Use manifest instead", value=False, key="mp_manifest")
    manifest_path = st.text_input(
        "Manifest YAML",
        value=str(DEFAULT_MANIFEST.relative_to(root())),
        key="mp_man",
        disabled=not use_manifest,
    )

    if st.button("Run Maestro PEX report", type="primary", key="mp_run"):
        r = root()
        od = r / out_dir
        cal_path = Path(cal_out) if cal_out.strip() else None
        try:
            if use_manifest:
                result = run_maestro_pex(
                    manifest_path=r / manifest_path,
                    out_dir=od,
                    write_calibration=(r / cal_path) if cal_path else None,
                    repo_root=r,
                )
            else:
                row = process_run_pair(
                    MaestroRunSpec(
                        marker="gui_run",
                        nopex_csv=r / nopex,
                        pex_csv=r / pex,
                        signal=signal,
                        sample_time_ns=float(sample_ns),
                    )
                )
                od.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(od / "maestro_pex_summary.csv", index=False)
                result = {"summary_csv": str(od / "maestro_pex_summary.csv"), "row": row}
            st.success("Maestro PEX report written.")
            st.json(result)
        except Exception as e:
            st.error(str(e))

    summary = root() / out_dir / "maestro_pex_summary.csv"
    if summary.is_file():
        st.subheader("Summary preview")
        st.dataframe(pd.read_csv(summary), use_container_width=True)

with tab_corner:
    render_hardware_profile_banner("pex_corner_proxy")
    st.markdown(
        "Upload or list multiple PEX CSVs at different corners (same `/OA_Charge` sample time). "
        "Exports a **corner-derived σ proxy** — not Monte Carlo statistics."
    )
    corner_paths = st.text_area(
        "PEX CSV paths (one per line: `corner_label,path`)",
        value="",
        placeholder="tt_27c,stuff_from_cadence/pex1.csv",
        key="cr_paths",
    )
    sample_corner_ns = st.number_input("Sample time (ns)", 0.0, 1e6, 200.0, key="cr_t")
    if st.button("Preview corner spread", key="cr_preview"):
        from hwa_cim.maestro_pex import MaestroCornerSpec

        specs: list[MaestroCornerSpec] = []
        for line in corner_paths.strip().splitlines():
            if not line.strip() or "," not in line:
                continue
            corner, path = line.split(",", 1)
            specs.append(
                MaestroCornerSpec(
                    marker="gui_corner",
                    corner=corner.strip(),
                    pex_csv=root() / path.strip(),
                    sample_time_ns=float(sample_corner_ns),
                )
            )
        if len(specs) < 2:
            st.warning("Need at least two corners (label,path) lines.")
        else:
            detail, proxy = process_corner_runs(specs)
            st.dataframe(detail, use_container_width=True)
            st.dataframe(proxy, use_container_width=True)

with tab_sur:
    render_hardware_profile_banner("surrogate_mc")
    st.markdown(
        """
**Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation)**

Parses wide VIVA exports from manual PDK delta-parameter sweeps. Sigma is estimated from
FF/SS corner deltas (÷3), **not** foundry-certified UMC Monte Carlo.

Outputs are thesis/plot artifacts — **not** the Phase 5 `input_code` noise profile schema.
"""
    )
    sc1, sc2, sc3 = st.columns(3)
    cap_csv = sc1.text_input(
        "Cap sweep CSV",
        value="stuff_from_cadence/manual_mc_2_var_cap_1.csv",
        key="sur_cap",
    )
    dvth_csv = sc2.text_input(
        "dvth0 sweep CSV",
        value="stuff_from_cadence/manual_mc_4_var_1.csv",
        key="sur_dvth",
    )
    sur_out = Path(sc3.text_input("Output base", value="results/surrogate_mc", key="sur_out"))
    sur_t = st.number_input("Sample time (ns)", 0.0, 1e6, 200.25, key="sur_t")

    if st.button("Run surrogate MC parser (both sweeps)", type="primary", key="sur_run"):
        from hwa_cim.surrogate_mc import run_surrogate_mc

        r = root()
        try:
            cap_r = run_surrogate_mc(
                r / cap_csv,
                r / sur_out / "cap_sweep",
                sample_time_ns=float(sur_t),
                variable_group="mom_cap_grid",
                marker="mom_cap_grid",
            )
            dvth_r = run_surrogate_mc(
                r / dvth_csv,
                r / sur_out / "dvth0_sweep",
                sample_time_ns=float(sur_t),
                variable_group="dvth0_grid",
                marker="dvth0_grid",
            )
            st.success("Phase 4.5 surrogate summaries written.")
            st.json({"cap_sweep": cap_r.get("summary"), "dvth0_sweep": dvth_r.get("summary")})
        except Exception as e:
            st.error(str(e))

    if st.button("Generate Phase 4.5 plots", key="sur_plots"):
        from hwa_cim.plots import run_surrogate_mc_plots

        try:
            paths = run_surrogate_mc_plots(
                surrogate_base=root() / sur_out,
                sample_time_ns=float(sur_t),
            )
            st.success(f"Wrote {len(paths)} plot(s) under results/plots/03_surrogate_mc/")
            for p in paths:
                st.caption(str(p))
        except Exception as e:
            st.error(str(e))

    cap_sum = root() / sur_out / "cap_sweep" / "surrogate_mc_summary.csv"
    if cap_sum.is_file():
        st.subheader("Cap sweep summary")
        st.dataframe(pd.read_csv(cap_sum), use_container_width=True)

with tab_cadence:
    render_hardware_profile_banner("cadence_surrogate_stress")
    st.markdown(
        """
**Phase 4.5 — Cadence-informed surrogate stress** (AgDR-0007)

Maps normalized `/OA_Charge` spread from a Phase 4.5 summary to **relative output noise**
in baseline eval and HWA training. Use this mode for the proxy recovery graph — not Phase 5 MC.
"""
    )
    from hwa_gui.wizard.actions import DEFAULT_SURROGATE_SUMMARY

    sum_path = st.text_input(
        "Surrogate summary CSV",
        value=DEFAULT_SURROGATE_SUMMARY,
        key="cad_sum",
    )
    full = root() / sum_path
    if full.is_file():
        try:
            from hwa_cim.cadence_stress import load_cadence_stress_profile

            prof = load_cadence_stress_profile(full)
            c1, c2, c3 = st.columns(3)
            c1.metric("mean_output (V)", f"{prof.mean_output:.6f}")
            c2.metric("sigma_output (V)", f"{prof.sigma_output:.6f}")
            c3.metric("σ_rel (applied stress)", f"{prof.surrogate_sigma_rel * 100:.4f}%")
            st.dataframe(pd.read_csv(full), use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("Summary missing — run **Phase 4.5 Surrogate MC** tab first.")
    st.caption(
        "Train/eval: **Run → HWA train** with profile "
        "**Phase 4.5 — Cadence-informed surrogate stress**, or the guided wizard with that profile."
    )
    plot_png = root() / "results/plots/05_cadence_stress/01_hwa_recovery_under_cadence_stress.png"
    if plot_png.is_file():
        st.subheader("Recovery plot")
        st.image(str(plot_png), use_container_width=True)

with tab_mc:
    render_hardware_profile_banner("monte_carlo_csv")
    st.markdown(
        """
Expected columns (case-insensitive): `input_code`, `ideal_output` (or `ideal`),
`mean_output` (or `mean`), `sigma` (or `std`). Optional: `CSNR_dB` / `csnr`.
"""
    )
    path_str = st.text_input("Path to CSV", value="", key="mc_path")
    up = st.file_uploader("Or upload a CSV", type=["csv"], key="mc_up")

    df_preview: pd.DataFrame | None = None
    path_used: Path | None = None

    if up is not None:
        df_preview = pd.read_csv(up)
        st.success("Uploaded file loaded for preview.")
        tmp = root() / "results" / ".dashboard_upload_noise.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(up.getbuffer())
        path_used = tmp
    elif path_str.strip():
        p = root() / path_str.strip()
        if p.exists():
            df_preview = pd.read_csv(p)
            path_used = p
        else:
            st.warning("File not found at that path (relative to project root).")

    if df_preview is not None:
        st.subheader("Preview")
        st.dataframe(df_preview.head(50), use_container_width=True)

    if path_used is not None and df_preview is not None:
        try:
            prof = NoiseProfileCSV.load(path_used)
            st.subheader("NoiseProfileCSV summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", len(prof.input_code))
            m2.metric("sigma_mean", f"{prof.sigma_mean:.6g}")
            m3.metric("sigma_max", f"{prof.sigma_max:.6g}")
            st.caption(f"Resolved path: `{prof.path}`")
        except Exception as e:
            st.error(f"Validation failed: {e}")

    st.caption(
        "Use **Run → 3a HWA train** with hardware profile **True Monte Carlo CSV**."
    )
