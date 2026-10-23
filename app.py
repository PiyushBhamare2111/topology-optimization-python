"""Streamlit front-end:  streamlit run app.py  ->  http://localhost:8501"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from topopt import Settings, make_case, optimize
from topopt.cases import CASES, DEFAULT_MESH
from topopt.export import export_stl, triangle_volume
from topopt.visualize import gif_bytes, plot_convergence, plot_density, plot_stress, render_stl

st.set_page_config(page_title="Topology Optimization in Python", page_icon=":material/architecture:", layout="wide")

ACCENT = "#ff8a3d"
st.markdown(
    f"""<style>
    .stApp {{ background-color: #0b0c0f; color: #e8e6e3; }}
    [data-testid="stSidebar"] {{ background-color: #14161b; }}
    h1, h2, h3 {{ color: #e8e6e3; }}
    .accent {{ color: {ACCENT}; }}
    div.stButton > button {{ background-color: {ACCENT}; color: #0b0c0f; border: 0; font-weight: 700; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
    </style>""",
    unsafe_allow_html=True,
)

LABELS = {"mbb": "MBB beam", "cantilever": "Cantilever", "lbracket": "L-bracket", "cclamp": "C-clamp (PLA, N/mm/MPa)"}

st.title("Topology Optimization in Python")
st.caption("2D SIMP compliance minimisation - Q4 plane-stress FEA, density filtering, optimality-criteria update.")

with st.sidebar:
    st.header("Load case")
    case = st.selectbox("Case", list(CASES), format_func=LABELS.get, key="case")
    dx, dy = DEFAULT_MESH[case]
    nelx = st.slider("Elements in x (nelx)", 20, 240, min(dx, 120), step=10, key=f"nelx_{case}")
    probe = make_case(case, nelx)
    st.caption(f"Mesh: {probe.nelx} x {probe.nely} = {probe.nelx * probe.nely:,} elements")
    st.header("Optimiser")
    volfrac = st.slider("Target volume fraction", 0.1, 0.9, float(probe.volfrac), step=0.001, format="%.3f", key=f"volfrac_{case}")
    penal = st.slider("SIMP penalty p", 1.0, 5.0, 3.0, step=0.5, key="penal")
    rmin = st.slider("Filter radius rmin [elements]", 1.1, 6.0, 1.5, step=0.1, key="rmin")
    filt = st.selectbox("Filter", ["density", "sensitivity", "heaviside"], key="filter")
    maxiter = st.slider("Max iterations", 10, 400, 150, step=10, key="maxiter")
    run = st.button("Run optimisation", type="primary", use_container_width=True)

st.markdown(f"**{probe.title}** - {probe.description or CASES[case].__doc__.strip().splitlines()[0]}")

if run:
    problem = make_case(case, nelx)
    settings = Settings(volfrac=volfrac, penal=penal, rmin=rmin, filter=filt, maxiter=maxiter)
    bar = st.progress(0.0, text="Starting...")

    def cb(it, x, c, vol, ch):
        bar.progress(min(1.0, it / maxiter), text=f"iteration {it}: c = {c:.4g}, volume = {vol:.3f}, change = {ch:.3f}")

    res = optimize(problem, settings, callback=cb)
    bar.progress(1.0, text=f"Done - {res.iterations} iterations in {res.runtime:.1f} s")
    with tempfile.TemporaryDirectory() as td:
        stl_path = Path(td) / f"{case}.stl"
        tris = export_stl(res, stl_path)
        stl = stl_path.read_bytes()
    st.session_state["result"] = res
    st.session_state["stl"] = stl
    st.session_state["tris"] = tris
    st.session_state["gif"] = gif_bytes(res)

res = st.session_state.get("result")
if res is None:
    st.info("Pick a case and parameters in the sidebar, then press **Run optimisation**.")
else:
    s = res.summary()
    u = res.problem.units
    cols = st.columns(5)
    cols[0].metric("Compliance", f"{s['compliance']:.4g}", help=f"F^T u  [{u['compliance']}]")
    cols[1].metric("Volume fraction", f"{s['volume']:.3f}")
    cols[2].metric("Iterations", s["iterations"], help="converged" if s["converged"] else "hit max iterations")
    cols[3].metric("Runtime", f"{s['runtime_s']:.1f} s")
    if "mass_g" in s:
        cols[4].metric("Mass (PLA)", f"{s['mass_g']:.2f} g", help=f"peak von Mises {s['max_von_mises']:.1f} MPa")
    else:
        cols[4].metric("Grayness", f"{s['grayness']:.3f}", help="mean 4x(1-x); 0 = crisp black/white")

    tab1, tab2, tab3, tab4 = st.tabs(["Design", "Convergence", "Stress", "3D / STL"])

    def show(fig):
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plot_density(res))
        with c2:
            st.image(st.session_state["gif"], caption="Optimisation history (animated)", use_container_width=True)
    with tab2:
        c1, _ = st.columns([3, 1])
        with c1:
            show(plot_convergence(res))
    with tab3:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plot_stress(res))
        with c2:
            st.markdown(f"Peak element von Mises stress: **{s['max_von_mises']:.3g}** {u['stress'] if u['stress'] != '-' else '(dimensionless)'}")
            if "safety_factor" in s:
                st.markdown(f"PLA yield ~{s['yield_strength']:g} MPa -> safety factor **{s['safety_factor']:.2f}** "
                            "(2D plane-stress estimate, linear elastic).")
            st.caption("Stress is evaluated at element centres of the SIMP-interpolated material; void (x < 0.5) is blanked.")
    with tab4:
        tris = st.session_state["tris"]
        c1, c2 = st.columns([3, 2])
        with c1:
            show(render_stl(tris, title="", figsize=(8, 5.5)))
        with c2:
            st.markdown(f"**{len(tris):,}** triangles, watertight extrusion")
            st.markdown(f"Enclosed volume: **{triangle_volume(tris):.4g}** {u['length'] + '^3' if u['length'] != '-' else '(element units)'}")
            st.caption("Marching squares on the density field (iso = 0.5) -> polygons -> extrusion. "
                       "Download below and open in any slicer.")

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("STL (extruded)", st.session_state["stl"], file_name=f"{res.problem.name}.stl", mime="model/stl")
    d2.download_button("Animation (GIF)", st.session_state["gif"], file_name=f"{res.problem.name}.gif", mime="image/gif")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["iteration", "compliance", "volume", "change"])
    for i, row in enumerate(zip(res.compliance, res.volume, res.change), 1):
        w.writerow([i, *row])
    d3.download_button("History (CSV)", buf.getvalue(), file_name=f"{res.problem.name}_history.csv", mime="text/csv")
    npy = io.BytesIO()
    np.save(npy, res.image)
    d4.download_button("Density (NPY)", npy.getvalue(), file_name=f"{res.problem.name}_density.npy")
    with st.expander("Run summary (JSON)"):
        st.code(json.dumps(s, indent=2), language="json")
