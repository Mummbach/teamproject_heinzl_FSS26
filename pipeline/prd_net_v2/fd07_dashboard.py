"""
PRD-Net v2 — fd07: Interactive Dashboard (Streamlit)
=====================================================
The full interactive dashboard over ALL test patients. Reads the fd06 exports
and reuses the figure builders from fd07_report.py.

Peers are training patients and the export names them without their values, so
selecting one additionally pulls the train feature matrix, cohort.csv and
y_train.parquet via fd07_report.load_peer_source().

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
import fd07_report as R   # reuse load() + figure builders (no streamlit dependency there)

st.set_page_config(page_title="PRD-Net v2 Dashboard", layout="wide")


@st.cache_data
def load(window):
    return R.load(window)


@st.cache_data
def load_peers(window):
    return R.load_peer_source(window)


def lab(v):
    return "Long Stay" if v == 1 else "Short Stay"


# The three exported peer lists, in display order.
PEER_TABLES = (
    ("top_similar_peers", "3 most similar patients"),
    ("top_long_peers",    "3 most similar long-stay"),
    ("top_short_peers",   "3 most similar short-stay"),
)


def peer_button_label(peer, src):
    """'35302525 · Short Stay · 2.1d' — the id stays first so the button reads
    as the patient, with the observed outcome and the concrete stay length that
    makes 'Long Stay' mean something."""
    los = float(src["cohort"].loc[peer["stay_id"], "los"])
    return f"{peer['stay_id']} · {lab(peer['true_label'])} · {los:.1f}d"


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
labels = list(options)

# Keep the same patient selected across window/filter switches. Option labels
# embed window-specific p=/pred/truth, so Streamlit can't match the previous
# selection by text alone and would otherwise silently jump back to the top
# of the list — track the stable stay_id instead.
stay_ids = [options[l]["stay_id"] for l in labels]
prev_stay_id = st.session_state.get("selected_stay_id")
default_idx = stay_ids.index(prev_stay_id) if prev_stay_id in stay_ids else 0

sel = st.sidebar.selectbox("Patient", labels, index=default_idx)
rec = options[sel]
st.session_state["selected_stay_id"] = rec["stay_id"]

# A peer belongs to the patient it was picked under. Drop the selection when
# either the patient or the window changes, otherwise the comparison below would
# show a peer of the previously selected patient.
if st.session_state.get("peer_owner") != (window, rec["stay_id"]):
    st.session_state["peer_owner"] = (window, rec["stay_id"])
    st.session_state["peer_active"] = None

top_n = st.sidebar.radio("Top features", R.TOP_N_LEVELS, horizontal=True,
                         help="How many of the highest-contribution features to show. "
                              "Capped by fd06's TOP_K at export time.")
peers = load_peers(window)

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
    st.plotly_chart(R.fig_importance(glob, top=top_n), use_container_width=True)

# ── Per-patient view: Patient -> Prediction -> Explanation -> Peer Group -> Delta ──

# 1. Patient — general info, descriptive only (not model input, except age)
st.markdown("### Patient")
st.caption(f"stay_id {rec['stay_id']}")
st.write(R.patient_info_line(rec))

# 2. Prediction
st.markdown("### Prediction")
correct = "✓ correct" if rec["pred_label"] == rec["true_label"] else "✗ wrong"
c1, c2, c3, c4 = st.columns(4)
c1.metric("p(Long Stay)", f"{rec['prob']:.3f}")
c2.metric("Prediction", lab(rec["pred_label"]))
c3.metric("Truth", lab(rec["true_label"]))
c4.metric("Result", correct)
if rec["wrong_reasons"]:
    st.error("**Why this prediction is probably wrong**\n"
             + "\n".join(f"- {r}" for r in rec["wrong_reasons"]))

# 3. Explanation — raw values first (readable without knowing what a "logit"
# is), then the model-internal contribution chart, then CXR corroboration.
st.markdown("### Explanation")
st.caption("Patient vs. prototypes (raw clinical values)")
feats = R.positionable_features(rec, top_n)
if feats:
    st.dataframe(pd.DataFrame([{
        "Feature": R.feat_label(f),
        "Patient": rec["patient"][f],
        "Long-stay proto": rec["long_prototype"][f],
        "Short-stay proto": rec["short_prototype"][f],
    } for f in feats]), hide_index=True, use_container_width=True)
    note = R.categorical_note(rec, top_n)
    if note:
        st.caption(note)
else:
    st.info(R.NO_POSITIONABLE_NOTE.format(n=top_n))
st.plotly_chart(R.fig_contributions(rec, top_n), use_container_width=True)
cs = rec["cxr_support"]
if cs["has_report"]:
    if cs["findings"]:
        names = ", ".join(R.feat_label(f["feature"]) for f in cs["findings"])
        st.success(f"Chest X-ray report also supports this: {names}.")
    else:
        st.caption("Chest X-ray report on file: no notable findings.")

# 4. Peer Group Comparison — who the peer group is (matches fd02's hard
# filter) + what happened to them, then the individual peer tables.
st.markdown("### Peer Group Comparison")
crit = R.peer_group_criteria_line(rec)
if crit: st.write(crit)
outcome = R.peer_group_outcome_line(rec)
if outcome: st.write(outcome)
s = rec["support"]
lf = " (global fallback)" if s["long_fallback"] else ""
sf = " (global fallback)" if s["short_fallback"] else ""
st.caption(f"Peer support (hard filter): {s['n_long_filtered']} long-stay{lf} · "
           f"{s['n_short_filtered']} short-stay{sf}")
st.caption("Click a patient to compare them against this one.")

# Buttons rather than a selectable dataframe: st.dataframe only registers a
# selection when its checkbox column is clicked, never the id cell itself, which
# is the thing a reader actually aims at. Buttons also make the click a discrete
# event, so there is no cross-table selection state to arbitrate.
active = st.session_state.get("peer_active")
for col, (field, title) in zip(st.columns(3), PEER_TABLES):
    col.markdown(f"**{title}**")
    for row, peer in enumerate(rec[field]):
        if col.button(peer_button_label(peer, peers),
                      key=f"peer_{field}_{row}_{window}_{rec['stay_id']}",
                      width="stretch",
                      type="primary" if active == (field, row) else "secondary"):
            active = (field, row)
            st.session_state["peer_active"] = active
            st.rerun()   # repaint so the clicked button shows as selected

if active:
    field, row = active
    peer_id = rec[field][row]["stay_id"]
    st.markdown(f"#### Peer {peer_id} vs. this patient")
    st.write(R.peer_info_line(peer_id, peers))
    st.dataframe(pd.DataFrame(R.peer_comparison_rows(rec, peer_id, peers, top_n)),
                 hide_index=True, use_container_width=True)
    st.caption("Peers are training patients, so their stay length is observed, not predicted.")

# 5. Delta / Difference — where the patient sits between the two prototypes
st.markdown("### Delta / Difference")
if R.positionable_features(rec, top_n):
    st.plotly_chart(R.fig_prototype_position(rec, top_n), use_container_width=True)
    note = R.categorical_note(rec, top_n)
    if note:
        st.caption(note)
else:
    st.info(R.NO_POSITIONABLE_NOTE.format(n=top_n))
