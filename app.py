import json
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from oceanembed import (
    DEPTHS,
    REGIONS,
    generate_surface_state,
    reconstruct_profile,
    estimate_thermocline,
    compute_ohc,
    uncertainty_estimate,
    physics_diagnostics,
    synthetic_argo_validation,
    feature_importance,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="OceanEmbed NIO | Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "sst": 28.0,
    "sss": 35.0,
    "ssh": 0.05,
    "u_cur": 0.10,
    "v_cur": 0.05,
    "u_wind": 3.0,
    "v_wind": 1.0,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

PRESETS = {
    "Custom": None,
    "☀️ Calm ocean": dict(sst=28.0, sss=35.0, ssh=0.02, u_cur=0.08, v_cur=0.03, u_wind=2.0, v_wind=0.8),
    "🔥 Marine heatwave": dict(sst=32.0, sss=34.7, ssh=0.25, u_cur=0.35, v_cur=0.20, u_wind=2.5, v_wind=1.0),
    "🌪️ Strong mixing": dict(sst=29.0, sss=35.2, ssh=-0.12, u_cur=0.45, v_cur=-0.35, u_wind=12.0, v_wind=7.0),
    "🌀 Cyclone-like mixing": dict(sst=30.0, sss=35.1, ssh=-0.30, u_cur=0.60, v_cur=-0.55, u_wind=-15.0, v_wind=12.0),
}

# ============================================================
# GLOBAL STYLING
# ============================================================
st.markdown(
    """
<style>
:root {
    --bg: #07111f;
    --panel: #0d1b2a;
    --panel2: #10253a;
    --line: #20384d;
    --text: #eaf6ff;
    --muted: #91a9bd;
    --cyan: #36d7ff;
    --blue: #4e8cff;
    --green: #4be3a2;
    --amber: #ffc857;
    --red: #ff6b6b;
}
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: rgba(7,17,31,0.92); }
[data-testid="stSidebar"] { background: #091522; border-right: 1px solid var(--line); }
.block-container { max-width: 1500px; padding-top: 1rem; padding-bottom: 2rem; }

h1, h2, h3 { color: var(--text) !important; }
.stMarkdown p, .stCaption { color: #a9bfd0; }
label, [data-testid="stWidgetLabel"] p { color: #cfe3f2 !important; }

.hero {
    position: relative;
    overflow: hidden;
    padding: 1.7rem 1.8rem;
    border: 1px solid #22506b;
    border-radius: 24px;
    background:
      radial-gradient(circle at 85% 15%, rgba(54,215,255,.24), transparent 28%),
      radial-gradient(circle at 10% 100%, rgba(78,140,255,.20), transparent 32%),
      linear-gradient(135deg, #0b2437 0%, #071521 58%, #0a1b2c 100%);
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
    margin-bottom: 1.1rem;
}
.hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -.03em; }
.hero p { margin: .35rem 0 0; color: #9fc1d7; max-width: 900px; }
.hero-badge {
    display: inline-block; margin-bottom: .65rem; padding: .28rem .7rem;
    border-radius: 999px; border: 1px solid #24627d; background: rgba(54,215,255,.08);
    color: #7de8ff; font-size: .75rem; font-weight: 700; letter-spacing: .08em;
}

.section-title { color: #dff6ff; font-size: 1.15rem; font-weight: 800; margin: .2rem 0 .7rem; }
.section-sub { color: #7893a8; font-size: .85rem; margin-top: -.45rem; margin-bottom: .8rem; }

.kpi {
    min-height: 116px; padding: 1rem 1.05rem; border-radius: 18px;
    border: 1px solid var(--line); background: linear-gradient(145deg,#102338,#0b1928);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03), 0 10px 30px rgba(0,0,0,.12);
}
.kpi .icon { font-size: 1.2rem; }
.kpi .label { color: #86a2b7; font-size: .78rem; font-weight: 700; margin-top: .35rem; }
.kpi .value { color: #f3fbff; font-size: 1.55rem; font-weight: 850; line-height: 1.15; margin-top: .2rem; }
.kpi .delta { color: #62e6b0; font-size: .72rem; margin-top: .28rem; }

.panel {
    border: 1px solid var(--line); background: #0b1928; border-radius: 20px;
    padding: 1rem 1.1rem; box-shadow: 0 12px 35px rgba(0,0,0,.12);
}
.panel-title { color: #dff6ff; font-weight: 800; font-size: 1rem; }
.panel-note { color: #7893a8; font-size: .76rem; }

.status {
    border-radius: 14px; padding: .75rem .9rem; border: 1px solid;
    font-weight: 700; margin: .35rem 0;
}
.status-green { background: rgba(75,227,162,.08); border-color: rgba(75,227,162,.35); color: #72edbb; }
.status-amber { background: rgba(255,200,87,.08); border-color: rgba(255,200,87,.35); color: #ffd775; }
.status-red { background: rgba(255,107,107,.08); border-color: rgba(255,107,107,.35); color: #ff9696; }
.status-blue { background: rgba(54,215,255,.08); border-color: rgba(54,215,255,.35); color: #83eaff; }

[data-testid="stTabs"] button { color: #9fb5c7; font-weight: 700; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #52ddff; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #eaf6ff !important; }
.stButton > button { border-radius: 12px; border: 1px solid #28465d; background: #10263a; color: #e8f8ff; }
.stButton > button:hover { border-color: #36d7ff; color: #ffffff; }
.stDownloadButton > button { border-radius: 12px; }

div[data-testid="stDataFrame"] { border: 1px solid #20384d; border-radius: 14px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def html_card(title, value, icon="🌊", subtitle=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="icon">{icon}</div>
            <div class="label">{title}</div>
            <div class="value">{value}</div>
            <div class="delta">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_start(title, note=""):
    note_html = f'<div class="panel-note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="panel"><div class="panel-title">{title}</div>{note_html}',
        unsafe_allow_html=True,
    )


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


def status_box(text, kind="blue"):
    st.markdown(f'<div class="status status-{kind}">{text}</div>', unsafe_allow_html=True)


def dark_layout(fig, height=450, margin=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#091725",
        font=dict(color="#dceef8"),
        height=height,
        margin=margin or dict(l=35, r=20, t=45, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#183044", zerolinecolor="#27445a")
    fig.update_yaxes(gridcolor="#183044", zerolinecolor="#27445a")
    return fig


def risk_label(sst_value, physics_score, mean_unc):
    if sst_value >= 31.0 or physics_score < 60:
        return "HIGH ATTENTION", "red"
    if sst_value >= 29.5 or mean_unc > 0.45 or physics_score < 75:
        return "WATCH", "amber"
    return "STABLE", "green"


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("# 🌊 OceanEmbed")
st.sidebar.caption("North Indian Ocean intelligence console")
st.sidebar.markdown("---")

region_name = st.sidebar.selectbox("Ocean region", list(REGIONS.keys()), index=0)
date = st.sidebar.date_input("Analysis date", value=pd.Timestamp("2026-09-12").date())
preset = st.sidebar.selectbox("Scenario preset", list(PRESETS.keys()))

if preset != "Custom" and st.sidebar.button("Apply scenario", use_container_width=True):
    for k, v in PRESETS[preset].items():
        st.session_state[k] = v
    st.rerun()

with st.sidebar.expander("🛰️ Surface observations", expanded=True):
    sst = st.slider("SST (°C)", 18.0, 34.0, step=0.1, key="sst")
    sss = st.slider("SSS (PSU)", 30.0, 38.0, step=0.1, key="sss")
    ssh = st.slider("SSH / SLA (m)", -0.8, 0.8, step=0.01, key="ssh")
    u_cur = st.slider("U current (m/s)", -1.5, 1.5, step=0.01, key="u_cur")
    v_cur = st.slider("V current (m/s)", -1.5, 1.5, step=0.01, key="v_cur")
    u_wind = st.slider("U wind (m/s)", -20.0, 20.0, step=0.1, key="u_wind")
    v_wind = st.slider("V wind (m/s)", -20.0, 20.0, step=0.1, key="v_wind")

with st.sidebar.expander("🧠 Model controls", expanded=True):
    window = st.select_slider("Temporal window (days)", [3, 5, 7, 14], value=7)
    attention = st.slider("Attention strength", 0.2, 1.0, 0.75, 0.05)
    extreme = st.slider("Extreme-event weight", 1.0, 5.0, 2.0, 0.1)
    runs = st.slider("Uncertainty runs", 10, 60, 30, 5)
    show_unc = st.checkbox("Show uncertainty band", True)

r1, r2 = st.sidebar.columns(2)
if r1.button("🎲 Random", use_container_width=True):
    rng = np.random.default_rng()
    for k, lo, hi, dec in [
        ("sst", 25, 32, 1), ("sss", 33.5, 36.5, 1), ("ssh", -0.4, 0.4, 2),
        ("u_cur", -0.8, 0.8, 2), ("v_cur", -0.8, 0.8, 2),
        ("u_wind", -15, 15, 1), ("v_wind", -15, 15, 1),
    ]:
        st.session_state[k] = round(float(rng.uniform(lo, hi)), dec)
    st.rerun()
if r2.button("🔄 Reset", use_container_width=True):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Prototype mode")
st.sidebar.caption("GLORYS training + independent ARGO validation are required for scientific deployment.")

# ============================================================
# MODEL / EMULATOR
# ============================================================
region = region_name
surface = generate_surface_state(
    region, pd.Timestamp(date), sst, sss, ssh, u_cur, v_cur, u_wind, v_wind, window
)
profile = reconstruct_profile(surface, attention, extreme)
thermocline = estimate_thermocline(DEPTHS, profile)
ohcc = compute_ohc(DEPTHS, profile)
unc = uncertainty_estimate(surface, attention, extreme, runs)
physics = physics_diagnostics(DEPTHS, profile, sst)
importance = feature_importance(surface)
confidence = max(50, min(99, 100 - unc.mean() * 35))
mean_unc = float(np.mean(unc))
risk, risk_kind = risk_label(sst, physics["score"], mean_unc)

# Extra derived indicators for dashboard exploration.
gradient = np.gradient(profile, DEPTHS)
current_speed = float(np.hypot(u_cur, v_cur))
wind_speed = float(np.hypot(u_wind, v_wind))
# A simple emulator-only mixing indicator; not a scientific oceanographic diagnostic.
mixing_index = float(np.clip(45 + wind_speed * 2.7 + current_speed * 22 + abs(ssh) * 35, 0, 100))
stratification_proxy = float(np.clip(100 - np.mean(np.abs(gradient)) * 900, 0, 100))
anomaly_index = float(np.clip((sst - 28.0) * 14 + (ssh * 35), -100, 100))

# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-badge">SIH26066 · NORTH INDIAN OCEAN · DAILY 0.25° CONCEPT</div>
      <h1>OceanEmbed — Subsurface Ocean Intelligence</h1>
      <p>Interactive satellite-to-subsurface temperature reconstruction console for exploring vertical thermal structure, uncertainty, physics diagnostics and regional behavior.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP STATUS ROW
# ============================================================
status_col1, status_col2, status_col3 = st.columns([1.4, 1, 1])
with status_col1:
    status_box(f"● {risk} · {region_name}", risk_kind)
with status_col2:
    status_box(f"📅 {date.strftime('%d %b %Y')} · {preset}", "blue")
with status_col3:
    status_box(f"🧪 Emulator confidence: {confidence:.0f}%", "green" if confidence >= 80 else "amber")

# ============================================================
# KPI GRID — 4 + 4, not cramped 5-card strip
# ============================================================
st.markdown('<div class="section-title">Mission snapshot</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    html_card("Thermocline depth", f"{thermocline:.1f} m", "🌡️", "Estimated transition layer")
with k2:
    html_card("OHC proxy", f"{ohcc:,.0f} MJ/m²", "🔥", "Integrated upper-ocean signal")
with k3:
    html_card("Mean uncertainty", f"±{mean_unc:.2f} °C", "📏", "Across reconstructed depths")
with k4:
    html_card("Physics score", f"{physics['score']:.1f}/100", "⚛️", "Consistency diagnostics")

st.write("")
k5, k6, k7, k8 = st.columns(4, gap="medium")
with k5:
    html_card("Confidence", f"{confidence:.0f}%", "🎯", "Prototype uncertainty score")
with k6:
    html_card("Surface SST", f"{sst:.1f} °C", "☀️", "Satellite-style input")
with k7:
    html_card("Mixing index", f"{mixing_index:.0f}/100", "🌪️", "Emulator diagnostic")
with k8:
    html_card("Anomaly index", f"{anomaly_index:+.0f}", "📈", "Relative surface anomaly")

st.caption("Prototype/emulator mode: the dashboard demonstrates the intended OceanEmbed workflow. It is not a GLORYS-trained scientific prediction product yet.")

# ============================================================
# NAVIGATION TABS
# ============================================================
tabs = st.tabs([
    "🏠 Mission Overview",
    "🌡️ Profile Lab",
    "🗺️ Ocean Field",
    "🧠 AI Explainability",
    "⚛️ Physics & Risk",
    "🎯 ARGO Validation",
    "🔬 Scenario Lab",
    "📦 Export & Research",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tabs[0]:
    left, right = st.columns([1.45, 1], gap="large")

    with left:
        panel_start("Vertical thermal structure", "Temperature and uncertainty across the requested depth levels")
        fig = go.Figure()
        if show_unc:
            fig.add_trace(go.Scatter(
                x=profile + unc, y=DEPTHS, line=dict(width=0), showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=profile - unc, y=DEPTHS, fill="tonexty",
                fillcolor="rgba(54,215,255,.12)", line=dict(width=0),
                name="Uncertainty"
            ))
        fig.add_trace(go.Scatter(
            x=profile, y=DEPTHS, mode="lines+markers", name="OceanEmbed",
            line=dict(color="#57dcff", width=4), marker=dict(size=7),
            customdata=np.column_stack([unc]),
            hovertemplate="Temperature %{x:.2f} °C<br>Depth %{y:.0f} m<br>±%{customdata[0]:.2f} °C<extra></extra>",
        ))
        fig.add_hline(y=thermocline, line_dash="dash", line_color="#ffc857",
                      annotation_text=f"Thermocline {thermocline:.0f} m")
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)")
        dark_layout(fig, 520)
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

    with right:
        panel_start("Surface forcing profile", "Seven satellite-style predictors driving the emulator")
        forcing = pd.DataFrame({
            "Feature": ["SST", "SSS", "SSH/SLA", "U current", "V current", "U wind", "V wind"],
            "Normalized": [
                np.clip((sst - 18) / 16, 0, 1), np.clip((sss - 30) / 8, 0, 1),
                np.clip((ssh + .8) / 1.6, 0, 1), np.clip((u_cur + 1.5) / 3, 0, 1),
                np.clip((v_cur + 1.5) / 3, 0, 1), np.clip((u_wind + 20) / 40, 0, 1),
                np.clip((v_wind + 20) / 40, 0, 1),
            ],
        })
        fig = go.Figure(go.Bar(
            x=forcing["Normalized"], y=forcing["Feature"], orientation="h",
            marker_color="#4e8cff", text=(forcing["Normalized"] * 100).round(0).astype(int).astype(str) + "%",
            textposition="outside"
        ))
        fig.update_xaxes(range=[0, 1.12], title="Normalized input")
        dark_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)
        panel_end()

        panel_start("Operational signal")
        if risk_kind == "green":
            status_box("Stable surface state. No strong emulator-level warning signal.", "green")
        elif risk_kind == "amber":
            status_box("Watch conditions. Elevated temperature, uncertainty or mixing signal detected.", "amber")
        else:
            status_box("High attention. Extreme surface forcing or low physics consistency is present.", "red")
        panel_end()

    st.markdown('<div class="section-title">Research pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5, gap="small")
    for col, icon, title, text in [
        (p1, "🛰️", "Surface inputs", "SST · SSS · SSH · currents · winds"),
        (p2, "🧩", "Embedding", "Spatial + temporal representation"),
        (p3, "🧠", "Reconstruction", "Depth-aware temperature decoder"),
        (p4, "⚛️", "Physics", "Gradient + surface consistency"),
        (p5, "🎯", "Validation", "Independent ARGO evaluation"),
    ]:
        with col:
            st.markdown(
                f'<div class="panel" style="min-height:115px"><div style="font-size:1.3rem">{icon}</div><b style="color:#eaf6ff">{title}</b><div class="panel-note">{text}</div></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# TAB 2 — PROFILE LAB
# ============================================================
with tabs[1]:
    st.markdown('<div class="section-title">Interactive profile laboratory</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        selected_depth = st.select_slider("Inspect depth", options=list(DEPTHS), value=100, key="profile_depth")
        temp = float(np.interp(selected_depth, DEPTHS, profile))
        sigma = float(np.interp(selected_depth, DEPTHS, unc))
        a, b, c = st.columns(3)
        with a: html_card("Selected depth", f"{selected_depth:.0f} m", "📍", "Profile cursor")
        with b: html_card("Temperature", f"{temp:.2f} °C", "🌡️", "Reconstructed")
        with c: html_card("Uncertainty", f"±{sigma:.2f} °C", "📏", "At selected depth")

        fig = go.Figure()
        if show_unc:
            fig.add_trace(go.Scatter(x=profile + unc, y=DEPTHS, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=profile - unc, y=DEPTHS, fill="tonexty", fillcolor="rgba(54,215,255,.14)", line=dict(width=0), name="± uncertainty"))
        fig.add_trace(go.Scatter(x=profile, y=DEPTHS, mode="lines+markers", name="Temperature", line=dict(color="#ff9fb2", width=4)))
        fig.add_trace(go.Scatter(x=[temp], y=[selected_depth], mode="markers", marker=dict(size=15, color="#ffc857", line=dict(width=2, color="#ffffff")), name="Selected depth"))
        fig.add_hline(y=thermocline, line_dash="dash", annotation_text=f"Thermocline {thermocline:.0f} m")
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)")
        dark_layout(fig, 620)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        panel_start("Depth response table", "Use the table to inspect the complete 15-level reconstruction")
        profile_df = pd.DataFrame({
            "Depth (m)": DEPTHS,
            "Temperature (°C)": np.round(profile, 3),
            "Uncertainty (°C)": np.round(unc, 3),
            "dT/dz (°C/m)": np.round(gradient, 5),
        })
        st.dataframe(profile_df, use_container_width=True, hide_index=True, height=530)
        panel_end()

# ============================================================
# TAB 3 — OCEAN FIELD
# ============================================================
with tabs[2]:
    st.markdown('<div class="section-title">North Indian Ocean spatial explorer</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        map_depth = st.select_slider("Map depth", options=list(DEPTHS), value=100, key="map_depth")
    with mc2:
        field_mode = st.selectbox("Field", ["Temperature", "Uncertainty", "Anomaly proxy"])
    with mc3:
        show_region_points = st.checkbox("Show regional markers", True)

    lon = np.linspace(45, 105, 121)
    lat = np.linspace(5, 30, 101)
    LON, LAT = np.meshgrid(lon, lat)
    dt = float(np.interp(map_depth, DEPTHS, profile))
    base = sst + 1.4 * np.sin((LON - 65) / 8) - 1.1 * np.cos((LAT - 17) / 5) + ssh * 2 + 0.8 * np.sin((LON + LAT) / 9)
    zfield = base - (sst - dt) * (0.55 + 0.35 * np.exp(-map_depth / 250))
    uncertainty_field = mean_unc * (0.75 + 0.4 * np.abs(np.sin((LON - 60) / 12)) * np.cos((LAT - 17) / 7) ** 2)
    anomaly_field = zfield - np.mean(zfield)
    field = zfield if field_mode == "Temperature" else uncertainty_field if field_mode == "Uncertainty" else anomaly_field

    colorscale = "Turbo" if field_mode == "Temperature" else "Viridis" if field_mode == "Uncertainty" else "RdBu_r"
    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=field, colorscale=colorscale,
        colorbar=dict(title="°C" if field_mode != "Uncertainty" else "±°C"),
        hovertemplate="Lon %{x:.2f}°<br>Lat %{y:.2f}°<br>Value %{z:.2f}<extra></extra>",
    ))
    if show_region_points:
        fig.add_trace(go.Scatter(
            x=[REGIONS[r]["lon"] for r in REGIONS],
            y=[REGIONS[r]["lat"] for r in REGIONS],
            text=list(REGIONS.keys()), mode="markers+text", textposition="top center",
            marker=dict(size=8, color="#ffffff", line=dict(color="#07111f", width=2)),
            name="Regions"
        ))
    fig.update_layout(title=f"{field_mode} field at {map_depth:.0f} m")
    fig.update_xaxes(title="Longitude")
    fig.update_yaxes(title="Latitude")
    dark_layout(fig, 620)
    st.plotly_chart(fig, use_container_width=True)
    status_box("Spatial field is an emulator visualization. Production should render the learned daily 0.25° North Indian Ocean field.", "blue")

# ============================================================
# TAB 4 — AI EXPLAINABILITY
# ============================================================
with tabs[3]:
    st.markdown('<div class="section-title">Model explainability console</div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        imp = importance.sort_values("Importance")
        fig = go.Figure(go.Bar(
            x=imp.Importance, y=imp.Feature, orientation="h",
            marker_color="#36d7ff",
            text=(imp.Importance * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_xaxes(title="Relative contribution", range=[0, max(.5, float(imp.Importance.max()) * 1.25)])
        dark_layout(fig, 500)
        fig.update_layout(title="Feature contribution proxy")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        panel_start("Attention controls", "Explore how changing model settings affects the emulator")
        att = st.slider("Attention strength", 0.2, 1.0, attention, 0.05, key="ex_att")
        ext = st.slider("Extreme-event weight", 1.0, 5.0, extreme, 0.1, key="ex_ext")
        alt_profile = reconstruct_profile(surface, att, ext)
        alt_tc = estimate_thermocline(DEPTHS, alt_profile)
        alt_ohc = compute_ohc(DEPTHS, alt_profile)
        aa, bb = st.columns(2)
        with aa: html_card("Thermocline", f"{alt_tc:.1f} m", "🧠", "Adjusted model")
        with bb: html_card("OHC proxy", f"{alt_ohc:,.0f}", "🔥", "Adjusted model")
        diff = alt_profile - profile
        fig2 = go.Figure(go.Scatter(x=diff, y=DEPTHS, mode="lines+markers", line=dict(color="#ffc857", width=3)))
        fig2.add_vline(x=0, line_dash="dash")
        fig2.update_yaxes(autorange="reversed", title="Depth (m)")
        fig2.update_xaxes(title="Temperature change (°C)")
        dark_layout(fig2, 360)
        st.plotly_chart(fig2, use_container_width=True)
        panel_end()

    panel_start("Production explainability roadmap")
    st.write("CBAM spatial/channel attention → temporal Transformer attention → SHAP / permutation importance → learned embedding visualization.")
    panel_end()

# ============================================================
# TAB 5 — PHYSICS & RISK
# ============================================================
with tabs[4]:
    st.markdown('<div class="section-title">Physics-aware diagnostics & operational risk</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4, gap="medium")
    with a: html_card("Surface consistency", f"{physics['surface_consistency']:.1f}%", "🌊", "SST/profile agreement")
    with b: html_card("Smoothness", f"{physics['smoothness']:.1f}%", "〰️", "Vertical stability")
    with c: html_card("Gradient realism", f"{physics['gradient_realism']:.1f}%", "📐", "Thermal gradient")
    with d: html_card("Overall physics", f"{physics['score']:.1f}/100", "⚛️", "Composite score")

    pleft, pright = st.columns([1.25, 1], gap="large")
    with pleft:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gradient, y=DEPTHS, mode="lines+markers", line=dict(color="#4be3a2", width=3), name="dT/dz"))
        fig.add_vline(x=0, line_dash="dot")
        fig.add_hline(y=thermocline, line_dash="dash", annotation_text="Thermocline")
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="dT/dz (°C/m)")
        dark_layout(fig, 520)
        st.plotly_chart(fig, use_container_width=True)
    with pright:
        panel_start("Risk dashboard")
        risk_value = {"green": 25, "amber": 60, "red": 90}[risk_kind]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_value,
            title={"text": "Attention index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff6b6b" if risk_kind == "red" else "#ffc857" if risk_kind == "amber" else "#4be3a2"},
                "steps": [
                    {"range": [0, 40], "color": "#102f2a"},
                    {"range": [40, 70], "color": "#3a321d"},
                    {"range": [70, 100], "color": "#3d2428"},
                ],
            },
        ))
        dark_layout(fig, 330)
        st.plotly_chart(fig, use_container_width=True)
        status_box(f"Current state: {risk}", risk_kind)
        st.write(f"Surface SST: **{sst:.1f} °C**  ·  Wind speed: **{wind_speed:.1f} m/s**  ·  Current speed: **{current_speed:.2f} m/s**")
        st.write(f"Stratification proxy: **{stratification_proxy:.1f}/100**")
        panel_end()

# ============================================================
# TAB 6 — ARGO VALIDATION
# ============================================================
with tabs[5]:
    st.markdown('<div class="section-title">Independent validation workspace</div>', unsafe_allow_html=True)
    argo = synthetic_argo_validation(profile, uncertainty=unc, seed=42)
    a, b, c, d = st.columns(4, gap="medium")
    with a: html_card("RMSE", f"{argo['rmse']:.2f} °C", "📉", "Synthetic reference")
    with b: html_card("MAE", f"{argo['mae']:.2f} °C", "📊", "Synthetic reference")
    with c: html_card("R²", f"{argo['r2']:.4f}", "🎯", "Synthetic reference")
    with d: html_card("Bias", f"{argo['bias']:+.2f} °C", "⚖️", "Mean signed error")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure(go.Scatter(
            x=argo["observed"], y=argo["predicted"], mode="markers",
            marker=dict(size=9, color="#36d7ff"), customdata=DEPTHS,
            hovertemplate="Depth %{customdata:.0f}m<br>Reference %{x:.2f}°C<br>Prediction %{y:.2f}°C<extra></extra>",
            name="Depth levels"
        ))
        mn = min(argo["observed"].min(), argo["predicted"].min())
        mx = max(argo["observed"].max(), argo["predicted"].max())
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(color="#ffc857", dash="dash"), name="1:1"))
        fig.update_xaxes(title="Reference temperature (°C)")
        fig.update_yaxes(title="Predicted temperature (°C)")
        dark_layout(fig, 520)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        residual = argo["predicted"] - argo["observed"]
        fig = go.Figure(go.Bar(x=DEPTHS, y=residual, marker_color=np.where(residual >= 0, "#ff8b9e", "#36d7ff")))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_xaxes(title="Depth (m)")
        fig.update_yaxes(title="Residual (°C)")
        dark_layout(fig, 520)
        st.plotly_chart(fig, use_container_width=True)
    status_box("Synthetic validation only. Replace this block with independent INCOIS LAS Gridded ARGO observations for the final SIH evaluation.", "amber")

# ============================================================
# TAB 7 — SCENARIO LAB
# ============================================================
with tabs[6]:
    st.markdown('<div class="section-title">Scenario comparison laboratory</div>', unsafe_allow_html=True)
    selected = st.multiselect(
        "Select scenarios",
        list(PRESETS.keys())[1:],
        default=["☀️ Calm ocean", "🔥 Marine heatwave", "🌪️ Strong mixing"],
        max_selections=4,
    )

    records = []
    fig = go.Figure()
    for scenario_name in selected:
        values = PRESETS[scenario_name]
        s = generate_surface_state(
            region, pd.Timestamp(date), values["sst"], values["sss"], values["ssh"],
            values["u_cur"], values["v_cur"], values["u_wind"], values["v_wind"], window
        )
        p = reconstruct_profile(s, attention, extreme)
        tc = estimate_thermocline(DEPTHS, p)
        oc = compute_ohc(DEPTHS, p)
        records.append({"Scenario": scenario_name, "Thermocline (m)": tc, "OHC proxy": oc, "SST (°C)": values["sst"]})
        fig.add_trace(go.Scatter(x=p, y=DEPTHS, mode="lines+markers", name=scenario_name))

    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    fig.update_xaxes(title="Temperature (°C)")
    dark_layout(fig, 560)
    st.plotly_chart(fig, use_container_width=True)

    scenario_df = pd.DataFrame(records)
    if not scenario_df.empty:
        st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)
        st.bar_chart(scenario_df.set_index("Scenario")[["Thermocline (m)", "SST (°C)"]], use_container_width=True)

    st.markdown("### Sensitivity sweep")
    sweep_feature = st.selectbox("Vary one surface variable", ["SST", "SSS", "SSH/SLA", "Wind speed"])
    if sweep_feature == "SST":
        sweep = np.linspace(max(18, sst - 3), min(34, sst + 3), 13)
    elif sweep_feature == "SSS":
        sweep = np.linspace(max(30, sss - 1.5), min(38, sss + 1.5), 13)
    elif sweep_feature == "SSH/SLA":
        sweep = np.linspace(max(-0.8, ssh - .35), min(.8, ssh + .35), 13)
    else:
        sweep = np.linspace(max(0, wind_speed - 8), min(28, wind_speed + 8), 13)

    out = []
    for value in sweep:
        vals = dict(sst=sst, sss=sss, ssh=ssh, u_cur=u_cur, v_cur=v_cur, u_wind=u_wind, v_wind=v_wind)
        if sweep_feature == "SST": vals["sst"] = float(value)
        elif sweep_feature == "SSS": vals["sss"] = float(value)
        elif sweep_feature == "SSH/SLA": vals["ssh"] = float(value)
        else:
            vals["u_wind"] = float(value)
            vals["v_wind"] = 0.0
        ss = generate_surface_state(region, pd.Timestamp(date), vals["sst"], vals["sss"], vals["ssh"], vals["u_cur"], vals["v_cur"], vals["u_wind"], vals["v_wind"], window)
        pp = reconstruct_profile(ss, attention, extreme)
        out.append({"Input": value, "Thermocline": estimate_thermocline(DEPTHS, pp), "OHC": compute_ohc(DEPTHS, pp)})
    sweep_df = pd.DataFrame(out)
    fig = px.line(sweep_df, x="Input", y=["Thermocline", "OHC"], markers=True)
    fig.update_layout(title=f"Sensitivity to {sweep_feature}")
    dark_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 8 — EXPORT & RESEARCH
# ============================================================
with tabs[7]:
    st.markdown('<div class="section-title">Experiment export & research mapping</div>', unsafe_allow_html=True)
    export_df = pd.DataFrame({
        "date": [str(date)] * len(DEPTHS),
        "region": [region_name] * len(DEPTHS),
        "depth_m": DEPTHS,
        "temperature_c": profile,
        "uncertainty_c": unc,
        "dT_dz_c_per_m": gradient,
    })
    config = {
        "date": str(date),
        "region": region_name,
        "scenario": preset,
        "window_days": window,
        "attention_strength": attention,
        "extreme_weight": extreme,
        "uncertainty_runs": runs,
        "surface_inputs": {
            "SST_C": sst, "SSS_PSU": sss, "SSH_m": ssh,
            "U_current_mps": u_cur, "V_current_mps": v_cur,
            "U_wind_mps": u_wind, "V_wind_mps": v_wind,
        },
        "outputs": {
            "thermocline_m": thermocline,
            "ohc_proxy_MJ_m2": ohcc,
            "mean_uncertainty_C": mean_unc,
            "physics_score": physics["score"],
            "confidence_percent": confidence,
            "risk_state": risk,
        },
        "prototype_note": "Emulator/demo output. Replace with GLORYS-trained model and independent INCOIS Gridded ARGO validation.",
    }

    a, b, c = st.columns(3)
    with a:
        st.download_button("⬇️ Temperature CSV", export_df.to_csv(index=False), "oceanembed_prediction.csv", "text/csv", use_container_width=True)
    with b:
        st.download_button("⬇️ Experiment JSON", json.dumps(config, indent=2), "oceanembed_experiment.json", "application/json", use_container_width=True)
    with c:
        st.download_button("⬇️ Model summary TXT", "OceanEmbed-NIO\n\n" + json.dumps(config, indent=2), "oceanembed_summary.txt", "text/plain", use_container_width=True)

    st.dataframe(export_df.round(4), use_container_width=True, hide_index=True)

    st.markdown("### SIH implementation roadmap")
    roadmap = pd.DataFrame([
        ["1", "Data ingestion", "Daily satellite fields on 0.25° grid", "Planned"],
        ["2", "Target generation", "GLORYS depth-wise temperature", "Planned"],
        ["3", "Embedding engine", "CNN / attention / ViT representation", "Prototype"],
        ["4", "Depth decoder", "15-level temperature reconstruction", "Prototype"],
        ["5", "Physics-aware loss", "Surface + vertical consistency", "Prototype"],
        ["6", "Uncertainty", "MC dropout / ensemble / quantiles", "Prototype"],
        ["7", "Independent validation", "INCOIS LAS Gridded ARGO", "Planned"],
        ["8", "Operational product", "Daily field + APIs + dashboard", "Planned"],
    ], columns=["Stage", "Module", "Implementation", "Status"])
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

    with st.expander("🔎 Technical notes"):
        st.write(""
            "This dashboard preserves the OceanEmbed prototype feature set while adding a mission overview, spatial field explorer, explainability controls, physics/risk view, sensitivity analysis, scenario comparison, richer exports and an SIH implementation roadmap. Current numerical outputs are generated by the included emulator, not a GLORYS-trained network."
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("OceanEmbed-NIO · SIH26066 · Interactive research/demo software · Daily 0.25° North Indian Ocean concept")
