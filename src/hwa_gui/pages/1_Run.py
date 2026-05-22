"""Configure and run pipeline jobs (background thread + live log)."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Run · HWA-CiM Lab",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_cim.maestro_pex import DEFAULT_PEX_CALIBRATION

from hwa_gui.components import (
    apply_page_style,
    confirm_overwrite,
    pipeline_quick_reference_md,
    render_hardware_profile_banner,
    render_pipeline_sidebar,
    root,
)
from hwa_gui.job_runner import get_job, start_job
from hwa_gui.log_panel import JOB_ID, render_live_log
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()
render_pipeline_sidebar(current="Run")

LOG_PATH = root() / "results" / ".dashboard_last.log"


def _maybe_start(fn_label: str, out_dir: Path, thunk) -> None:
    j = get_job(JOB_ID)
    if j and j.running:
        st.warning("A job is already running. Wait for it to finish.")
        return
    if not confirm_overwrite(out_dir):
        st.info("Enable the overwrite confirmation to proceed.")
        return
    st.session_state["log_path"] = LOG_PATH
    start_job(JOB_ID, LOG_PATH, thunk)
    st.success(f"Started: {fn_label}")
    st.rerun()


def _job_banner() -> None:
    jj = get_job(JOB_ID)
    if not jj:
        return
    if jj.running:
        st.warning(
            "**Job running** — only one job at a time on this page. "
            "The live log is at the **bottom**; it auto-refreshes while the job runs."
        )
    elif jj.done and jj.error:
        st.error("Last job ended with an error — see traceback at the bottom.")
    elif jj.done:
        st.success("**Last job finished.** Scroll to **Run log** below for full output.")


st.title("Run pipeline")
st.caption("Tabs follow the usual thesis order. Each tab is a natural stop: outputs go to disk under `results/`.")
_job_banner()

st.info(
    """
**How this page works**

- Pick the **tab** that matches the experiment you want (phase numbers match the thesis / README).
- Adjust paths if you use non-default output folders (defaults match the sidebar ✓ hints).
- Press the **primary** button to start; confirm overwrite if the output folder already has files.
- **One job at a time** — if something is running, wait for it before launching another.
"""
)

with st.expander("Phases, partial runs, and picking up where you left off", expanded=False):
    st.markdown(
        """
Each tab calls the same Python entrypoints as the CLI. If you already have `best.pt` or `metrics.json`
from an earlier session, **skip ahead** to the tab that needs those paths—nothing is stored only in the browser.

If you change output directories, use **Results** to locate artifacts; the sidebar checklist only tracks **default** paths.
"""
    )
    st.markdown(pipeline_quick_reference_md())

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "1 · Baseline",
        "2a · Noisy eval",
        "2b · Gamma sweep",
        "3a · HWA train",
        "3b · HWA sweep",
        "4 · Distill",
        "Fig · Thesis bars",
        "Fig · Parasitic",
    ]
)

with tab1:
    st.subheader("Phase 1 — Baseline")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** trains FP32 MNIST micro-MLP, INT4 PTQ checks, MAC parity; writes `best.pt` and `metrics.json`.

**Needs:** MNIST under your **Data dir** (downloaded automatically if missing).

**Metrics:** newer runs include `int4_ptq_test_accuracy_ideal` and `int4_ptq_test_accuracy_hardware`; older runs may only show `int4_ptq_test_accuracy` (see AgDR-0001).
"""
        )
    c1, c2 = st.columns(2)
    data_dir = Path(c1.text_input("Data dir", value="data", key="b_data"))
    out_dir = Path(c2.text_input("Output dir", value="results/run_baseline", key="b_out"))
    c3, c4, c5 = st.columns(3)
    epochs = c3.number_input("Epochs", 1, 500, 20, key="b_ep")
    batch = c4.number_input("Batch size", 8, 1024, 128, key="b_bs")
    lr = c5.number_input("Learning rate", 1e-6, 1.0, 1e-3, format="%.6f", key="b_lr")
    c6, c7 = st.columns(2)
    seed = c6.number_input("Seed", 0, 2**31 - 1, 42, key="b_seed")
    device = c7.selectbox("Device", ["cpu", "cuda"], index=0, key="b_dev")
    if st.button("Run baseline", type="primary", key="b_run"):
        od = root() / out_dir

        def _go() -> None:
            from hwa_cim.train_baseline import run_baseline

            run_baseline(
                data_dir=root() / data_dir,
                out_dir=od,
                epochs=int(epochs),
                batch_size=int(batch),
                lr=float(lr),
                seed=int(seed),
                device=device,
            )

        _maybe_start("baseline", od, _go)

with tab2:
    st.subheader("Phase 2 — Noisy eval")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** evaluates a trained checkpoint under CiM-shaped noise; optional JSON for the thesis middle bar.

**Needs:** a **Checkpoint** path (typically Phase 1 `best.pt`).
"""
        )
    ck = st.text_input("Checkpoint", value="results/run_baseline/best.pt", key="n_ck")
    c1, c2, c3 = st.columns(3)
    gamma = c1.number_input("gamma", 0.0, 1.0, 0.02, format="%.4f", key="n_g")
    seeds = c2.number_input("Seeds", 1, 100, 10, key="n_seeds")
    device = c3.selectbox("Device", ["cpu", "cuda"], key="n_dev")
    out_json = st.text_input(
        "Output JSON (optional)",
        value="results/run_baseline/noisy_eval.json",
        key="n_out",
        help="Set a path to save noisy eval stats for the thesis three-bar chart.",
    )
    data_dir = st.text_input("Data dir", value="data", key="n_data")
    if st.button("Run noisy eval", type="primary", key="n_run"):
        outp = Path(out_json) if out_json.strip() else None

        def _go() -> None:
            from hwa_cim.evaluate import run_noisy_eval

            run_noisy_eval(
                checkpoint=root() / ck,
                data_dir=root() / data_dir,
                gamma=float(gamma),
                seeds=int(seeds),
                device=device,
                out=(root() / outp) if outp else None,
            )

        _maybe_start("noisy eval", (root() / ck).parent, _go)

with tab3:
    st.subheader("Phase 2 — Gamma sweep")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** sweeps noise strength γ on a fixed checkpoint; writes CSV/JSON under the sweep output dir.

**Needs:** Phase 1 **Checkpoint**. Use **Charts → Gamma sweep** to visualize `gamma_sweep.csv` when done.
"""
        )
    ck = st.text_input("Checkpoint", value="results/run_baseline/best.pt", key="g_ck")
    out_dir = st.text_input("Sweep output dir", value="results/phase2_sweep", key="g_out")
    c1, c2 = st.columns(2)
    data_dir = c1.text_input("Data dir", value="data", key="g_data")
    device = c2.selectbox("Device", ["cpu", "cuda"], key="g_dev")
    if st.button("Run gamma sweep", type="primary", key="g_run"):
        od = root() / out_dir

        def _go() -> None:
            from hwa_cim.evaluate import run_sweep_gamma

            run_sweep_gamma(
                checkpoint=root() / ck,
                data_dir=root() / data_dir,
                device=device,
                out_dir=od,
            )

        _maybe_start("gamma sweep", od, _go)

with tab4:
    st.subheader("Phase 3 — HWA train")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** noise-aware training with a **hardware profile** (synthetic, Maestro PEX calibration, corner proxy, or Phase 5 MC CSV).

**Needs:** MNIST data. For **Maestro PEX**, run **Hardware profiles → Maestro PEX** first. For **MC CSV**, validate on **Hardware profiles → Monte Carlo CSV**.
"""
        )
    profile_labels = [
        "Synthetic AFM Noise",
        "Maestro PEX Calibration",
        "PEX Corner Proxy",
        "True Monte Carlo CSV",
    ]
    profile_to_mode = {
        "Synthetic AFM Noise": "synthetic",
        "Maestro PEX Calibration": "maestro_pex",
        "PEX Corner Proxy": "pex_corner_proxy",
        "True Monte Carlo CSV": "monte_carlo_csv",
    }
    hw_profile_label = st.selectbox(
        "Hardware profile mode",
        profile_labels,
        index=0,
        key="h_profile",
    )
    hw_mode = profile_to_mode[hw_profile_label]
    render_hardware_profile_banner(hw_mode)

    c1, c2 = st.columns(2)
    data_dir = Path(c1.text_input("Data dir", value="data", key="h_data"))
    out_dir = Path(c2.text_input("Output dir", value="results/run_hwa", key="h_out"))
    c3, c4 = st.columns(2)
    epochs = c3.number_input("Epochs", 1, 500, 40, key="h_ep")
    batch = c4.number_input("Batch size", 8, 1024, 128, key="h_bs")
    c5, c6, c7 = st.columns(3)
    lr = c5.number_input("Learning rate", 1e-6, 1.0, 1e-3, format="%.6f", key="h_lr")
    gamma = c6.number_input("gamma", 0.0, 1.0, 0.02, format="%.4f", key="h_g")
    alpha = c7.number_input("alpha (clip)", 0.1, 20.0, 3.0, format="%.2f", key="h_a")
    c8, c9 = st.columns(2)
    seed = c8.number_input("Seed", 0, 2**31 - 1, 42, key="h_seed")
    eval_seeds = c9.number_input("Eval noisy seeds", 1, 50, 10, key="h_es")
    c10, c11 = st.columns(2)
    device = c10.selectbox("Device", ["cpu", "cuda"], key="h_dev")
    cal_yaml = c11.text_input(
        "Calibration YAML",
        value="config/calibration.yaml",
        key="h_cal",
        help="Maestro PEX mode defaults to config/calibration_pex.yaml when left as schematic default.",
    )
    noise_profile = st.text_input(
        "Noise / corner profile CSV",
        value="",
        key="h_np",
        help="MC CSV for True Monte Carlo; corner_proxy_noise_profile.csv for PEX Corner Proxy.",
    )
    if hw_mode == "maestro_pex":
        st.caption(
            f"Uses synthetic noise + PEX calibration. Generate `{DEFAULT_PEX_CALIBRATION.name}` "
            "via **Hardware profiles → Maestro PEX** or `hwa-maestro-pex --write-calibration`."
        )
    if st.button("Run HWA train", type="primary", key="h_run"):
        od = root() / out_dir
        np_path = Path(noise_profile) if noise_profile.strip() else None
        r = root()

        if hw_mode == "monte_carlo_csv" and not np_path:
            st.error("True Monte Carlo CSV mode requires a noise profile CSV path.")
            st.stop()
        if hw_mode == "pex_corner_proxy":
            if not np_path:
                np_path = r / "results/maestro_pex/corner_proxy_noise_profile.csv"
            if not np_path.is_file():
                st.error(
                    "PEX Corner Proxy needs corner_proxy_noise_profile.csv — "
                    "generate corners on **Hardware profiles → PEX corners** first."
                )
                st.stop()

        if hw_mode == "synthetic":
            noise_mode = "synthetic"
            cal_path = r / cal_yaml if cal_yaml.strip() else None
        elif hw_mode == "maestro_pex":
            noise_mode = "synthetic"
            cal_path = r / DEFAULT_PEX_CALIBRATION
            if not cal_path.is_file() and cal_yaml.strip():
                cal_path = r / cal_yaml
        elif hw_mode == "pex_corner_proxy":
            noise_mode = "csv"
            cal_path = r / cal_yaml if cal_yaml.strip() else None
        else:
            noise_mode = "csv"
            cal_path = r / cal_yaml if cal_yaml.strip() else None

        def _go() -> None:
            from hwa_cim.train_hwa import run_hwa_train

            run_hwa_train(
                data_dir=r / data_dir,
                out_dir=od,
                epochs=int(epochs),
                batch_size=int(batch),
                lr=float(lr),
                gamma=float(gamma),
                alpha=float(alpha),
                seed=int(seed),
                device=device,
                noise_mode=noise_mode,
                noise_profile=(r / np_path) if np_path else None,
                calibration_yaml=cal_path,
                eval_noisy_seeds=int(eval_seeds),
                hardware_profile_mode=hw_mode,
            )

        _maybe_start("HWA train", od, _go)

with tab5:
    st.subheader("Phase 3 — HWA sweep (gamma × alpha grid)")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** grid search over γ and α; **long** runtime. Outputs CSV for **Charts → HWA sweep**.

**Needs:** MNIST data only (no checkpoint from Phase 1 required for this entrypoint).
"""
        )
    c1, c2 = st.columns(2)
    data_dir = Path(c1.text_input("Data dir", value="data", key="hs_data"))
    out_dir = Path(c2.text_input("Output dir", value="results/sweep_hwa", key="hs_out"))
    c3, c4, c5 = st.columns(3)
    epochs = c3.number_input("Epochs per cell", 1, 200, 30, key="hs_ep")
    batch = c4.number_input("Batch size", 8, 1024, 128, key="hs_bs")
    lr = c5.number_input("Learning rate", 1e-6, 1.0, 1e-3, format="%.6f", key="hs_lr")
    c6, c7 = st.columns(2)
    seed = c6.number_input("Seed", 0, 2**31 - 1, 42, key="hs_seed")
    device = c7.selectbox("Device", ["cpu", "cuda"], key="hs_dev")
    if st.button("Run HWA sweep", type="primary", key="hs_run"):
        od = root() / out_dir

        def _go() -> None:
            from hwa_cim.train_hwa import run_hwa_sweep

            run_hwa_sweep(
                data_dir=root() / data_dir,
                out_dir=od,
                epochs=int(epochs),
                batch_size=int(batch),
                lr=float(lr),
                seed=int(seed),
                device=device,
            )

        _maybe_start("HWA sweep", od, _go)

with tab6:
    st.subheader("Phase 4 — Distillation")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** teacher–student distillation with noisy student path (optional teacher checkpoint).

**Needs:** MNIST data. Optional **Teacher checkpoint**; leave empty if the run trains its own teacher.
"""
        )
    c1, c2 = st.columns(2)
    data_dir = Path(c1.text_input("Data dir", value="data", key="d_data"))
    out_dir = Path(c2.text_input("Output dir", value="results/run_distill", key="d_out"))
    c3, c4 = st.columns(2)
    te = c3.number_input("Teacher epochs", 1, 200, 30, key="d_te")
    se = c4.number_input("Student epochs", 1, 200, 40, key="d_se")
    c5, c6 = st.columns(2)
    batch = c5.number_input("Batch size", 8, 1024, 128, key="d_bs")
    lr = c6.number_input("Learning rate", 1e-6, 1.0, 1e-3, format="%.6f", key="d_lr")
    c7, c8, c9 = st.columns(3)
    gamma = c7.number_input("gamma", 0.0, 1.0, 0.02, format="%.4f", key="d_g")
    aclip = c8.number_input("alpha_clip", 0.1, 20.0, 3.0, format="%.2f", key="d_ac")
    dalpha = c9.number_input("distill_alpha (KL weight)", 0.0, 1.0, 0.7, format="%.2f", key="d_da")
    c10, c11 = st.columns(2)
    temp = c10.number_input("Temperature", 0.5, 20.0, 4.0, format="%.2f", key="d_temp")
    seed = c11.number_input("Seed", 0, 2**31 - 1, 42, key="d_seed")
    tck = st.text_input("Teacher checkpoint (optional)", value="", key="d_tck")
    device = st.selectbox("Device", ["cpu", "cuda"], key="d_dev")
    if st.button("Run distill", type="primary", key="d_run"):
        od = root() / out_dir
        tcp = Path(tck) if tck.strip() else None

        def _go() -> None:
            from hwa_cim.train_distill import run_distill

            run_distill(
                data_dir=root() / data_dir,
                out_dir=od,
                teacher_epochs=int(te),
                student_epochs=int(se),
                batch_size=int(batch),
                lr=float(lr),
                gamma=float(gamma),
                alpha_clip=float(aclip),
                distill_alpha=float(dalpha),
                temperature=float(temp),
                teacher_checkpoint=(root() / tcp) if tcp else None,
                seed=int(seed),
                device=device,
            )

        _maybe_start("distill", od, _go)

with tab7:
    st.subheader("Thesis — three-bar chart (PNG)")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** writes a static PNG comparing baseline / noisy / HWA bars.

**Needs:** `metrics.json` in the **baseline dir**; HWA **`best.pt`** with `metrics.json` beside it; optional **noisy eval JSON** for the middle bar (Phase 2a default path works if you saved it there).
"""
        )
    bdir = st.text_input("Baseline dir (metrics.json)", value="results/run_baseline", key="p_bdir")
    hw_ck = st.text_input("HWA checkpoint", value="results/run_hwa/best.pt", key="p_hw")
    noisy = st.text_input("Noisy eval JSON (optional)", value="results/run_baseline/noisy_eval.json", key="p_ne")
    out_png = st.text_input("Output PNG", value="results/figures/thesis_bars.png", key="p_out")
    if st.button("Generate thesis chart", type="primary", key="p_run"):
        od = root() / Path(out_png).parent
        ne_path = root() / noisy if noisy.strip() else None

        def _go() -> None:
            from hwa_cim.plots import run_thesis_chart

            run_thesis_chart(
                baseline_dir=root() / bdir,
                hwa_checkpoint=root() / hw_ck,
                noisy_eval_json=ne_path,
                out=root() / out_png,
            )

        _maybe_start("thesis chart", od, _go)

with tab8:
    st.subheader("Parasitic sweep figure (PNG)")
    with st.expander("What this does & prerequisites", expanded=False):
        st.markdown(
            """
**Does:** generates the parasitic comparison figure used in the thesis write-up (no training required).

**Needs:** nothing on disk — only the **PDK marker ratio** slider.
"""
        )
    out_png = st.text_input("Output PNG", value="results/figures/parasitic_sweep.png", key="par_out")
    pdk = st.number_input("PDK marker ratio", 0.0, 0.5, 0.30, format="%.2f", key="par_pdk")
    if st.button("Generate parasitic plot", type="primary", key="par_run"):
        od = root() / Path(out_png).parent

        def _go() -> None:
            from hwa_cim.plots import run_parasitic_plot

            run_parasitic_plot(out=root() / out_png, pdk_marker=float(pdk))

        _maybe_start("parasitic plot", od, _go)

st.divider()
st.subheader("Run log")
lp = st.session_state.get("log_path")
if lp:
    render_live_log(Path(lp))
else:
    st.caption("No job started yet — logs appear here after you press a run button.")

j = get_job(JOB_ID)
if j and j.done and j.error:
    st.error(j.error)
