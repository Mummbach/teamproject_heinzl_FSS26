"""
PRD-Net v2 — fd07: Standalone Dashboard Report (Plotly HTML)
============================================================
A first, self-contained dashboard *approach*. It reads ONLY the fd06 exports
(no model, no torch) and writes one interactive HTML file that opens in any
browser — no server, no extra install (uses plotly, already available).

It demonstrates the contrastive, per-feature story we want to show later:
  - Global overview: headline metrics, global feature importance, and the
    distance-view-vs-difference-view comparison.
  - Per-patient (dropdown over a curated set of illustrative cases):
      * prediction vs. truth,
      * "why" = per-feature contributions to the logit (red -> long, blue -> short),
      * where the patient sits BETWEEN the two prototypes per feature,
      * the 3 most-similar / 3 long-stay / 3 short-stay peers with their outcome.

The Streamlit app (fd07_dashboard.py) is the full interactive version over all
patients; this static report is the zero-dependency preview.

Run AFTER: fd06_dashboard-export.py.  Output: exports/fd_dashboard_{w}.html
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent))
import config_fd as C

LONG_C, SHORT_C, PAT_C = "#d62728", "#1f77b4", "#111111"   # red / blue / black

# Display-only relabeling for feature names that read as misleading/unclear to
# a clinician. Underlying column names (config_fd.CXR_FEATURES / feature_matrix
# columns) are left unchanged so exports, code, and cached parquet files don't
# need to move — only the label shown in charts/tables here is remapped.
#   "ventilator"   — column actually flags ET-tube/intubation mentions
#                    (DEVICE_TERMS in 01d_extract_radiology_features.py), not
#                    the literal word "ventilator".
#   "central_line" — column name is already accurate ICU terminology.
FEATURE_LABELS = {
    "ventilator": "Intubated (ET tube mentioned)",
    "central_line": "Central line",
}


def feat_label(feature):
    return FEATURE_LABELS.get(feature, feature)


def load(window):
    base = C.EXPORT_DIR / f"fd_explanations_test_{window}h.json"
    with open(base) as f:
        records = json.load(f)
    with open(C.EXPORT_DIR / f"fd_global_{window}h.json") as f:
        glob = json.load(f)
    return records, glob


def curated(records, n_each=4):
    """Pick illustrative cases: confident-correct, false-neg, false-pos."""
    def grp(t, p): return [r for r in records if r["true_label"] == t and r["pred_label"] == p]
    cl = sorted(grp(1, 1), key=lambda r: -r["prob"])[:n_each]
    cs = sorted(grp(0, 0), key=lambda r:  r["prob"])[:n_each]
    fn = sorted(grp(1, 0), key=lambda r:  r["prob"])[:2]   # missed long stays
    fp = sorted(grp(0, 1), key=lambda r: -r["prob"])[:2]   # false alarms
    return cl + cs + fn + fp


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_importance(glob, top=15):
    items = glob["feature_importance"][:top][::-1]
    fig = go.Figure(go.Bar(
        x=[d["importance"] for d in items], y=[feat_label(d["feature"]) for d in items],
        orientation="h", marker_color="#555"))
    fig.update_layout(
        title="Global feature importance (mean |contribution|, Δpos+Δneg summed)",
        height=460, margin=dict(l=160, r=20, t=50, b=40),
        xaxis_title="mean |contribution| (logit)")
    return fig


def fig_contributions(rec):
    tc = rec["top_contributions"][::-1]
    xs = [d["contribution"] for d in tc]
    ys = [f"{feat_label(d['feature'])}  (vs {d['prototype']})" for d in tc]
    colors = [LONG_C if x > 0 else SHORT_C for x in xs]
    text = [f"Δ={d['raw_delta']:+g}" for d in tc]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=colors,
                           text=text, textposition="outside", cliponaxis=False))
    fig.update_layout(
        title="Why: per-feature contribution to the logit  (red → long stay, blue → short stay)",
        height=360, margin=dict(l=230, r=60, t=50, b=40),
        xaxis_title="Contribution w·Δ (logit)")
    fig.add_vline(x=0, line_color="#999")
    return fig


def fig_prototype_position(rec):
    """Per top-feature: where the patient sits between short (0) and long (1) proto."""
    tc = rec["top_contributions"]
    feats = list(dict.fromkeys(d["feature"] for d in tc))[:6][::-1]   # unique, keep order
    short_x, long_x, pat_x, rows = [], [], [], []
    for f in feats:
        lo, sh, pa = rec["long_prototype"][f], rec["short_prototype"][f], rec["patient"][f]
        denom = (lo - sh) or 1e-9
        pos = (pa - sh) / denom
        short_x.append(0.0); long_x.append(1.0)
        pat_x.append(min(1.5, max(-0.5, pos)))   # clamp so outliers don't squash the 0–1 region
        rows.append(feat_label(f))
    fig = go.Figure()
    for x, name, col, sym in [(short_x, "Short-stay prototype", SHORT_C, "circle"),
                              (long_x, "Long-stay prototype", LONG_C, "circle"),
                              (pat_x, "Patient", PAT_C, "diamond")]:
        fig.add_trace(go.Scatter(x=x, y=rows, mode="markers", name=name,
                                 marker=dict(color=col, size=12, symbol=sym)))
    fig.add_vline(x=0, line_dash="dot", line_color=SHORT_C)
    fig.add_vline(x=1, line_dash="dot", line_color=LONG_C)
    fig.update_layout(
        title="Patient position between prototypes (0 = short stay, 1 = long stay; clamped to ±0.5)",
        height=320, margin=dict(l=180, r=30, t=50, b=40),
        xaxis=dict(title="normalized: 0 = short-stay prototype, 1 = long-stay prototype",
                   range=[-0.6, 1.6]))
    return fig


# ── HTML assembly ─────────────────────────────────────────────────────────────

def _lab(v): return "Long Stay" if v == 1 else "Short Stay"


# Plain-text formatters shared with fd07_dashboard.py (imported there as `R`)
# so the two renderers describe a patient identically instead of each
# re-deriving the same strings from `rec`.

def patient_info_line(rec):
    """'Age 67 · Female · Neuro Intermediate · EW EMER. admission · Dx: Respiratory'.
    Descriptive only, not a model input (age is the one exception)."""
    pi = rec["patient_info"]
    age = rec["patient"].get("age")
    parts = []
    if age is not None: parts.append(f"Age {age:.0f}")
    parts.append(pi["gender"])
    if pi["icu_unit"]: parts.append(pi["icu_unit"])
    if pi["admission_type"]: parts.append(f"{pi['admission_type']} admission")
    if pi["diagnosis_category"]: parts.append(f"Diagnosis Category: {pi['diagnosis_category']}")
    return " · ".join(parts)


def peer_group_criteria_line(rec):
    """What fd02's hard filter actually matched peers on for this patient
    (one-hot buckets — can read 'not filtered' where patient_info_line shows
    real text, e.g. an ICU type outside the 7 recognized buckets)."""
    age = rec["patient"].get("age")
    if age is None:
        return ""
    fc = rec["peer_group"]["filter_criteria"]
    lo, hi = age - C.AGE_TOLERANCE, age + C.AGE_TOLERANCE
    icu = fc["icu_unit"] or "not filtered — no matching ICU category"
    adm = fc["admission_type"] or "not filtered — no matching admission category"
    dx = fc["diagnosis_category"] or "not filtered — no matching diagnosis category"
    return (f"Same diagnosis category ({dx}), same ICU type ({icu}), "
            f"same admission type ({adm}), age {lo:.0f}–{hi:.0f}")


def peer_group_outcome_line(rec):
    """Observed outcomes among the peer group, or '' if unavailable."""
    pg = rec["peer_group"]
    if pg["avg_los_days"] is None:
        return ""
    return (f"Among {pg['n_peers']} matching peers: avg LOS {pg['avg_los_days']:.1f} days, "
            f"{pg['pct_long_stay']:.0f}% long-stay")


def patient_info_html(rec):
    return f"<div class='patinfo'>{patient_info_line(rec)}</div>"


def peer_group_html(rec):
    crit, outcome = peer_group_criteria_line(rec), peer_group_outcome_line(rec)
    return (f"<div class='peergroup'><b>Peer group composition</b><br>{crit}"
            + (f"<br>{outcome}" if outcome else "") + "</div>")


def peers_html(rec):
    def tbl(title, peers):
        rows = "".join(
            f"<tr><td>{p['stay_id']}</td><td>{_lab(p['true_label'])}</td></tr>" for p in peers)
        return (f"<div class='peers'><b>{title}</b>"
                f"<table><tr><th>stay_id</th><th>Outcome</th></tr>{rows}</table></div>")
    return ("<div class='peerwrap'>"
            + tbl("3 most similar patients", rec["top_similar_peers"])
            + tbl("3 most similar long-stay", rec["top_long_peers"])
            + tbl("3 most similar short-stay", rec["top_short_peers"])
            + "</div>")


def support_html(rec):
    s = rec["support"]
    lf = " (global fallback)" if s["long_fallback"] else ""
    sf = " (global fallback)" if s["short_fallback"] else ""
    return (f"<p class='support'>Peer support (hard filter): "
            f"{s['n_long_filtered']} long-stay{lf} · {s['n_short_filtered']} short-stay{sf}</p>")


def raw_values_html(rec):
    """Patient vs. prototypes in raw clinical units (readable without knowing what a logit is)."""
    feats = list(dict.fromkeys(d["feature"] for d in rec["top_contributions"]))[:8]
    rows = "".join(
        f"<tr><td>{feat_label(f)}</td><td>{rec['patient'][f]}</td>"
        f"<td>{rec['long_prototype'][f]}</td><td>{rec['short_prototype'][f]}</td></tr>"
        for f in feats)
    return ("<div class='rawvals'><b>Patient vs. prototypes (raw clinical values)</b>"
            "<table><tr><th>Feature</th><th>Patient</th><th>Long-stay proto</th>"
            f"<th>Short-stay proto</th></tr>{rows}</table></div>")


def cxr_support_html(rec):
    """Corroborating 'anamnesis' line from the CXR report — kept separate from
    top_contributions since CXR only covers ~2% of patients and a coverage
    flag (has_cxr_report) isn't itself a finding a clinician can read."""
    cs = rec["cxr_support"]
    if not cs["has_report"]:
        return ""
    if not cs["findings"]:
        return "<p class='cxr'>Chest X-ray report on file: no notable findings.</p>"
    names = ", ".join(feat_label(f["feature"]) for f in cs["findings"])
    return f"<p class='cxr'>Chest X-ray report also supports this: {names}.</p>"


def reasons_html(rec):
    if not rec["wrong_reasons"]:
        return ""
    items = "".join(f"<li>{r}</li>" for r in rec["wrong_reasons"])
    return (f"<div class='wrong'><b>Why this prediction is probably wrong</b>"
            f"<ul>{items}</ul></div>")


def _sec(title, body):
    return f"<div class='sectionlabel'>{title}</div>{body}"


def patient_block(rec, idx):
    correct = "✓ correct" if rec["pred_label"] == rec["true_label"] else "✗ wrong"
    stay_header = f"<div class='phead'>stay_id <b>{rec['stay_id']}</b></div>"
    pred_header = (f"<div class='phead'>p(Long Stay) = <b>{rec['prob']:.3f}</b> &nbsp;|&nbsp; "
                   f"Prediction: <b>{_lab(rec['pred_label'])}</b> &nbsp;|&nbsp; "
                   f"Truth: <b>{_lab(rec['true_label'])}</b> &nbsp;|&nbsp; {correct}</div>")
    contrib_fig = fig_contributions(rec).to_html(full_html=False, include_plotlyjs=False)
    position_fig = fig_prototype_position(rec).to_html(full_html=False, include_plotlyjs=False)

    display = "block" if idx == 0 else "none"
    sections = (
        _sec("Patient", stay_header + patient_info_html(rec)),
        _sec("Prediction", pred_header + reasons_html(rec)),
        _sec("Explanation", raw_values_html(rec) + contrib_fig + cxr_support_html(rec)),
        _sec("Peer Group Comparison", peer_group_html(rec) + support_html(rec) + peers_html(rec)),
        _sec("Delta / Difference", position_fig),
    )
    return (f"<div class='patient' id='pat{idx}' style='display:{display}'>"
            + "".join(sections) + "</div>")


def build(window):
    records, glob = load(window)
    cur = curated(records)
    m = glob["metrics"]
    metrics_html = (
        "<table class='m'><tr><th>Metric</th><th>Value</th></tr>"
        + "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in m.items())
        + f"<tr><td>prevalence</td><td>{glob['prevalence']:.4f}</td></tr>"
        + f"<tr><td>threshold</td><td>{glob['threshold']:.2f}</td></tr></table>")
    dvd = (f"<p class='note'>Distance view (||Δpos||&lt;||Δneg||): F1 = "
           f"<b>{glob['distance_view_f1']:.3f}</b> &nbsp;vs&nbsp; "
           f"difference model: F1 = <b>{glob['difference_view_f1']:.3f}</b>. "
           "Only the difference view names the responsible features.</p>")

    options = "".join(
        f"<option value='{i}'>{r['stay_id']} — Prediction {_lab(r['pred_label'])} / "
        f"Truth {_lab(r['true_label'])} (p={r['prob']:.2f})</option>"
        for i, r in enumerate(cur))
    patients = "".join(patient_block(r, i) for i, r in enumerate(cur))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PRD-Net v2 Dashboard ({window}h)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#222;max-width:1000px}}
 h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd}}
 table.m,.peers table,.rawvals table{{border-collapse:collapse}}
 .m td,.m th,.peers td,.peers th,.rawvals td,.rawvals th{{border:1px solid #ccc;padding:3px 10px;font-size:13px}}
 .rawvals{{margin:8px 0}}
 .phead{{background:#f4f4f4;padding:8px 12px;border-radius:6px;margin:8px 0;font-size:14px}}
 .peerwrap{{display:flex;gap:24px;flex-wrap:wrap;margin:6px 0 30px}}
 .note{{background:#fff7e6;border-left:3px solid #f0a020;padding:8px 12px;font-size:14px}}
 .wrong{{background:#fdecea;border-left:3px solid #d62728;padding:8px 12px;font-size:14px;margin:8px 0}}
 .wrong ul{{margin:6px 0 0 18px}} .support{{color:#555;font-size:13px;margin:4px 0}}
 .cxr{{background:#eef7f0;border-left:3px solid #2e8b57;padding:8px 12px;font-size:14px;margin:8px 0}}
 .sectionlabel{{text-transform:uppercase;letter-spacing:.04em;font-size:12px;font-weight:700;color:#888;margin:22px 0 6px;border-top:1px solid #eee;padding-top:14px}}
 .patient>.sectionlabel:first-child{{border-top:none;margin-top:6px;padding-top:0}}
 .patinfo{{font-size:14px;color:#333;margin:2px 0 4px}}
 .peergroup{{background:#f4f4f4;border-radius:6px;padding:8px 12px;font-size:14px;margin:6px 0}}
 select{{font-size:14px;padding:4px}}
</style></head><body>
<h1>PRD-Net v2 — Feature Difference Dashboard ({window}h)</h1>
<h2>Global overview (test, n={glob['n_test']})</h2>
{metrics_html}{dvd}
{fig_importance(glob).to_html(full_html=False, include_plotlyjs=False)}
<h2>Patient view (curated examples)</h2>
<label>Select patient: <select id="sel" onchange="showPatient()">{options}</select></label>
{patients}
<script>
function showPatient(){{
  var n={len(cur)}, v=document.getElementById('sel').value;
  for(var i=0;i<n;i++){{document.getElementById('pat'+i).style.display=(''+i===v)?'block':'none';}}
  window.dispatchEvent(new Event('resize'));
}}
</script>
<p class='note'>Note: static approach with curated cases. The full interactive version
over all {glob['n_test']} patients is the Streamlit app (fd07_dashboard.py).</p>
</body></html>"""

    C.EXPORT_DIR.mkdir(exist_ok=True)
    out = C.EXPORT_DIR / f"fd_dashboard_{window}h.html"
    out.write_text(html, encoding="utf-8")
    return out, len(cur)


if __name__ == "__main__":
    W = C.WINDOW_HOURS
    print(f"fd07 — standalone HTML report (window={W})")
    out, n = build(W)
    print(f"  {n} curated patients -> {out}")
    print(f"  Open in browser:  open {out}")
