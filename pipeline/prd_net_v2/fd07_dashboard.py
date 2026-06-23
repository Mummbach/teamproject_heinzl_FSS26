"""
PRD-Net v2 — fd07: Interactive Dashboard (Streamlit)
=====================================================
The full interactive dashboard over ALL test patients. Reads only the fd06
exports and reuses the figure builders from fd07_report.py.

Requires streamlit (not in the base env):
    pip install streamlit
Run:
    streamlit run fd07_dashboard.py
(Running it with plain `python` auto-relaunches under streamlit.)

For a zero-install preview use fd07_report.py (static HTML, curated cases).

Run AFTER: fd06_dashboard-export.py.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Auto-relaunch under `streamlit run` if started with plain `python` (e.g. the
# IDE Run button). Otherwise streamlit runs in "bare mode" and prints a
# "missing ScriptRunContext" warning for every st.* call (~50 lines).
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    raise SystemExit(stcli.main())

sys.path.append(str(Path(__file__).parent))
import config_fd as C
import fd07_report as R   # reuse load() + figure builders (no streamlit dependency there)

st.set_page_config(page_title="PRD-Net v2 Dashboard", layout="wide")


@st.cache_data
def load(window):
    return R.load(window)


def lab(v):
    return "Long Stay" if v == 1 else "Short Stay"


def peer_df(peers):
    return pd.DataFrame([{"stay_id": p["stay_id"], "Outcome": lab(p["true_label"])} for p in peers])


# ── Sidebar: window, filter, patient ──────────────────────────────────────────
st.sidebar.title("PRD-Net v2")
window = st.sidebar.radio("Observation window", [48, 24], horizontal=True)
records, glob = load(window)

filters = {
    "All": lambda r: True,
    "Correct": lambda r: r["pred_label"] == r["true_label"],
    "Wrong": lambda r: r["pred_label"] != r["true_label"],
    "Predicted Long Stay": lambda r: r["pred_label"] == 1,
    "Predicted Short Stay": lambda r: r["pred_label"] == 0,
    "False Negatives (missed long stay)": lambda r: r["true_label"] == 1 and r["pred_label"] == 0,
    "False Positives (false alarm)": lambda r: r["true_label"] == 0 and r["pred_label"] == 1,
}
fname = st.sidebar.selectbox("Filter", list(filters))
subset = [r for r in records if filters[fname](r)]
subset.sort(key=lambda r: -r["prob"])
st.sidebar.caption(f"{len(subset)} of {len(records)} patients")

if not subset:
    st.warning("No patients for this filter.")
    st.stop()

options = {f"{r['stay_id']} — pred {lab(r['pred_label'])} / true {lab(r['true_label'])} "
           f"(p={r['prob']:.2f})": r for r in subset}
sel = st.sidebar.selectbox("Patient", list(options))
rec = options[sel]

# ── Global overview ───────────────────────────────────────────────────────────
st.title(f"Feature Difference Dashboard ({window}h)")
with st.expander("Global overview (test)", expanded=False):
    m = glob["metrics"]
    cols = st.columns(len(m))
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k.upper(), f"{v:.3f}")
    st.info(f"Distance view (‖Δpos‖<‖Δneg‖): F1 = {glob['distance_view_f1']:.3f}  ·  "
            f"difference model: F1 = {glob['difference_view_f1']:.3f}. "
            "Only the difference view names the responsible features.")
    st.plotly_chart(R.fig_importance(glob), use_container_width=True)

# ── Per-patient view ──────────────────────────────────────────────────────────
correct = "✓ correct" if rec["pred_label"] == rec["true_label"] else "✗ wrong"
c1, c2, c3, c4 = st.columns(4)
c1.metric("p(Long Stay)", f"{rec['prob']:.3f}")
c2.metric("Prediction", lab(rec["pred_label"]))
c3.metric("Truth", lab(rec["true_label"]))
c4.metric("Result", correct)

# Why probably wrong (only for misclassified patients) + peer support
if rec["wrong_reasons"]:
    st.error("**Why this prediction is probably wrong**\n"
             + "\n".join(f"- {r}" for r in rec["wrong_reasons"]))
s = rec["support"]
lf = " (global fallback)" if s["long_fallback"] else ""
sf = " (global fallback)" if s["short_fallback"] else ""
st.caption(f"Peer support (hard filter): {s['n_long_filtered']} long-stay{lf} · "
           f"{s['n_short_filtered']} short-stay{sf}")

st.plotly_chart(R.fig_contributions(rec), use_container_width=True)
st.plotly_chart(R.fig_prototype_position(rec), use_container_width=True)

st.subheader("Most similar patients (training set, with outcome)")
p1, p2, p3 = st.columns(3)
p1.markdown("**3 most similar patients**");   p1.dataframe(peer_df(rec["top_similar_peers"]), hide_index=True)
p2.markdown("**3 most similar long-stay**");  p2.dataframe(peer_df(rec["top_long_peers"]), hide_index=True)
p3.markdown("**3 most similar short-stay**"); p3.dataframe(peer_df(rec["top_short_peers"]), hide_index=True)

with st.expander("Raw values: patient vs. prototypes (largest deviations)"):
    feats = list(dict.fromkeys(d["feature"] for d in rec["top_contributions"]))[:8]
    st.dataframe(pd.DataFrame([{
        "Feature": f,
        "Patient": rec["patient"][f],
        "Long-stay proto": rec["long_prototype"][f],
        "Short-stay proto": rec["short_prototype"][f],
    } for f in feats]), hide_index=True, use_container_width=True)
