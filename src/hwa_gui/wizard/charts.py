"""Wizard result panels — metrics cards and thesis figures."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_gui.components import hardware_profile_badge_from_metrics


def show_baseline_metrics_cards(metrics: dict) -> None:
    fp32 = metrics.get("fp32_test_accuracy")
    ideal = metrics.get("int4_ptq_test_accuracy_ideal")
    hw = metrics.get("int4_ptq_test_accuracy_hardware")
    legacy = metrics.get("int4_ptq_test_accuracy")

    c1, c2, c3 = st.columns(3)
    if fp32 is not None:
        c1.metric("FP32 test accuracy", f"{float(fp32) * 100:.2f}%")
    if ideal is not None:
        c2.metric("INT4 PTQ (ideal MAC)", f"{float(ideal) * 100:.2f}%")
    elif legacy is not None:
        c2.metric("INT4 PTQ", f"{float(legacy) * 100:.2f}%")
    if hw is not None:
        c3.metric("INT4 PTQ (hardware-shaped)", f"{float(hw) * 100:.2f}%")
    elif legacy is not None and ideal is None:
        c3.caption("Legacy metrics — see AgDR-0001 for dual INT4 keys.")


def show_noisy_crash_metrics(
    baseline_metrics: dict,
    noisy_eval: dict,
) -> None:
    fp32 = float(baseline_metrics.get("fp32_test_accuracy", 0.0))
    noisy_mean = float(noisy_eval.get("mean_accuracy", 0.0))
    gamma = noisy_eval.get("gamma", "—")

    c1, c2, c3 = st.columns(3)
    c1.metric("FP32 (clean)", f"{fp32 * 100:.2f}%")
    c2.metric(f"Noisy eval (γ={gamma})", f"{noisy_mean * 100:.2f}%")
    delta_pp = (noisy_mean - fp32) * 100
    c3.metric("Change", f"{delta_pp:+.2f} pp")

    st.caption(
        "Step 3 uses CiM-shaped noise on the baseline checkpoint. "
        "The thesis story is: naive noisy inference can diverge from clean accuracy; "
        "HWA in Step 4 retrains to recover robustness."
    )


def show_hwa_recovery_metrics(hwa_metrics: dict, noisy_eval: dict | None) -> None:
    badge = hardware_profile_badge_from_metrics(hwa_metrics)
    st.markdown(f"**Hardware profile:** `{badge}`")
    if hwa_metrics.get("profile_warning"):
        st.caption(str(hwa_metrics["profile_warning"]))

    hwa_noisy = float(hwa_metrics.get("final_noisy_mean", 0.0))
    c1, c2 = st.columns(2)
    c1.metric("HWA noisy test accuracy", f"{hwa_noisy * 100:.2f}%")
    if noisy_eval:
        prev = float(noisy_eval.get("mean_accuracy", 0.0))
        c2.metric("Step 3 noisy (same γ)", f"{prev * 100:.2f}%")
        c2.caption(f"Δ vs Step 3: {(hwa_noisy - prev) * 100:+.2f} pp")


def show_thesis_three_bar(
    repo: Path,
    *,
    baseline_dir: Path,
    hwa_ckpt: Path,
    noisy_json: Path | None,
) -> None:
    from hwa_gui.plotly_charts import figure_thesis_three_bar

    fig = figure_thesis_three_bar(
        baseline_dir=baseline_dir,
        hwa_checkpoint=hwa_ckpt,
        noisy_eval_json=noisy_json,
    )
    st.plotly_chart(fig, use_container_width=True)


def show_surrogate_mc_plots(repo: Path) -> None:
    plot_dir = repo / "results/plots/03_surrogate_mc"
    if not plot_dir.is_dir():
        st.info(
            "Phase 4.5 plots not found. Run **Refresh Phase 4.5 plots** below or use "
            "Advanced lab → Hardware profiles → Phase 4.5 Surrogate MC."
        )
        return
    pngs = sorted(plot_dir.glob("*.png"))
    if not pngs:
        st.info("No PNGs under `results/plots/03_surrogate_mc/`.")
        return
    for png in pngs[:4]:
        st.image(str(png), caption=png.name, use_container_width=True)


def show_pex_context_plot(repo: Path) -> None:
    candidates = [
        repo / "results/plots/02_hwa_context/01_hwa_vs_pex_context.png",
        repo / "results/plots/02_hwa_context/02_noisy_mnist_pex_three_bar.png",
    ]
    shown = False
    for p in candidates:
        if p.is_file():
            st.image(str(p), caption=p.name, use_container_width=True)
            shown = True
    if not shown:
        summary = repo / "results/maestro_pex/maestro_pex_summary.csv"
        if summary.is_file():
            from hwa_gui.plotly_charts import figure_maestro_summary_csv

            st.plotly_chart(
                figure_maestro_summary_csv(summary),
                use_container_width=True,
            )
        else:
            st.info(
                "No PEX context plot yet. Run Maestro PEX in "
                "**Advanced lab → Hardware profiles** or use existing Cadence CSVs."
            )


def show_cadence_stress_recovery(
    repo: Path,
    *,
    baseline_dir: Path,
    hwa_ckpt: Path,
) -> None:
    """Cadence-informed three-bar recovery (AgDR-0007)."""
    baseline_m = baseline_dir / "metrics.json"
    stress_json = repo / "results/run_baseline/noisy_eval_cadence_stress.json"
    hwa_m = hwa_ckpt.parent / "metrics.json"
    plot_png = (
        repo / "results/plots/05_cadence_stress/01_hwa_recovery_under_cadence_stress.png"
    )

    missing = [
        p for p in (baseline_m, stress_json, hwa_m) if not p.is_file()
    ]
    if missing:
        st.warning(
            "Need baseline metrics, `noisy_eval_cadence_stress.json` (Step 3), and HWA metrics. "
            f"Missing: {', '.join(p.name for p in missing)}"
        )
        return

    try:
        from hwa_cim.plots import run_cadence_stress_recovery_plot

        out = run_cadence_stress_recovery_plot(
            baseline_metrics=baseline_m,
            stress_eval_json=stress_json,
            hwa_stress_metrics=hwa_m,
            out_path=plot_png,
        )
        st.image(str(out), use_container_width=True)
        st.caption(
            "Phase 4.5; not foundry-certified Monte Carlo or final Phase 5. "
            "Proxy claim: HWA recovers under Cadence-informed surrogate stress."
        )
    except Exception as e:
        st.error(str(e))


def show_maestro_overlay_if_csvs_exist(repo: Path) -> None:
    nopex = repo / "stuff_from_cadence/nopex1.csv"
    pex = repo / "stuff_from_cadence/pex1.csv"
    if nopex.is_file() and pex.is_file():
        from hwa_gui.plotly_charts import figure_maestro_oa_charge_overlay

        st.plotly_chart(
            figure_maestro_oa_charge_overlay(nopex, pex),
            use_container_width=True,
        )
