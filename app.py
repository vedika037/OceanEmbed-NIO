import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

st.set_page_config(
    page_title="OceanEmbed-NIO",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.main {background: #f6f9fc;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg,#063970,#0b7285);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {margin:0 0 .35rem 0; font-size:2.25rem;}
.hero p {margin:.2rem 0; opacity:.92;}
.card {
    background:white; border:1px solid #e6edf3; border-radius:14px;
    padding:1rem; box-shadow:0 2px 10px rgba(0,0,0,.04);
}
.small {font-size:.86rem; color:#64748b;}
.badge {
    display:inline-block; padding:.25rem .55rem; border-radius:999px;
    background:#e6f7f7; color:#075985; font-size:.8rem; margin-right:.3rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🌊 OceanEmbed-NIO</h1>
<p><b>Physics-aware spatiotemporal satellite embedding for daily subsurface temperature reconstruction</b></p>
<p>North Indian Ocean • 5°N–30°N • 45°E–105°E • 0.25° prototype grid</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Experiment controls")

date = st.sidebar.date_input(
    "Analysis date",
    value=pd.Timestamp("2026-09-12").date()
)

region_name = st.sidebar.selectbox(
    "Region",
    list(REGIONS.keys()),
    index=0
)

region = region_name

st.sidebar.subheader("Surface inputs")
sst = st.sidebar.slider("SST (°C)", 18.0, 34.0, 28.0, 0.1)
sss = st.sidebar.slider("SSS (PSU)", 30.0, 38.0, 35.0, 0.1)
ssh = st.sidebar.slider("SSH anomaly (m)", -0.8, 0.8, 0.05, 0.01)
u_cur = st.sidebar.slider("U current (m/s)", -1.5, 1.5, 0.10, 0.01)
v_cur = st.sidebar.slider("V current (m/s)", -1.5, 1.5, 0.05, 0.01)
u_wind = st.sidebar.slider("U wind (m/s)", -20.0, 20.0, 3.0, 0.1)
v_wind = st.sidebar.slider("V wind (m/s)", -20.0, 20.0, 1.0, 0.1)

st.sidebar.subheader("Model controls")
window = st.sidebar.selectbox("Temporal window", [3, 5, 7, 14], index=2)
attention_strength = st.sidebar.slider("Attention strength", 0.2, 1.0, 0.75, 0.05)
uncertainty_runs = st.sidebar.slider("MC-Dropout runs", 10, 60, 30, 5)
extreme_weight = st.sidebar.slider("Extreme-event loss weight", 1.0, 5.0, 2.0, 0.1)

run = st.sidebar.button("🚀 Run OceanEmbed", use_container_width=True, type="primary")

# ---------- State ----------
surface = generate_surface_state(
    region=region,
    date=pd.Timestamp(date),
    sst=sst, sss=sss, ssh=ssh,
    u_cur=u_cur, v_cur=v_cur, u_wind=u_wind, v_wind=v_wind,
    window=window,
)

profile = reconstruct_profile(
    surface,
    attention_strength=attention_strength,
    extreme_weight=extreme_weight,
)
thermocline = estimate_thermocline(DEPTHS, profile)
ohc = compute_ohc(DEPTHS, profile)
unc = uncertainty_estimate(
    surface, attention_strength, extreme_weight, n_runs=uncertainty_runs
)
physics = physics_diagnostics(DEPTHS, profile, sst)
importance = feature_importance(surface)

if "has_run" not in st.session_state:
    st.session_state.has_run = True
if run:
    st.session.has_run = True

# ---------- Top KPIs ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Thermocline", f"{thermocline:.1f} m")
k2.metric("OHC proxy", f"{ohc:,.0f} MJ/m²")
k3.metric("Mean uncertainty", f"±{unc.mean():.2f} °C")
k4.metric("Physics score", f"{physics['score']:.1f}/100")
k5.metric("Confidence", f"{max(50, min(99, 100-unc.mean()*35)):.0f}%")

st.caption(
    "Prototype inference is interactive and deterministic. It demonstrates the complete "
    "OceanEmbed workflow using an ocean-state emulator; connect GLORYS/ARGO and trained "
    "weights for scientific production use."
)

# ---------- Main tabs ----------
tabs = st.tabs([
    "🌡️ Temperature Profile",
    "🗺️ Spatial Embedding",
    "🧠 Attention & Explainability",
    "⚛️ Physics Diagnostics",
    "🎯 ARGO Validation",
    "📊 Model & Research",
])

with tabs[0]:
    left, right = st.columns([1.6, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=profile, y=DEPTHS, mode="lines+markers",
            name="OceanEmbed prediction",
            line=dict(width=4),
        ))
        fig.add_trace(go.Scatter(
            x=profile + unc, y=DEPTHS, mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=profile - unc, y=DEPTHS, mode="lines",
            fill="tonexty", fillcolor="rgba(30,136,229,0.12)",
            line=dict(width=0), name="Uncertainty band",
        ))
        fig.add_hline(y=thermocline, line_dash="dash", annotation_text="Thermocline")
        fig.update_yaxes(autorange="reversed", title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)")
        fig.update_layout(height=560, margin=dict(l=30,r=20,t=20,b=30))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        table = pd.DataFrame({
            "Depth (m)": DEPTHS,
            "Temp (°C)": np.round(profile, 2),
            "± Uncertainty": np.round(unc, 2),
        })
        st.dataframe(table, use_container_width=True, height=560, hide_index=True)

with tabs[1]:
    st.subheader("North Indian Ocean surface-to-subsurface spatial state")
    lon = np.linspace(45, 105, 61)
    lat = np.linspace(5, 30, 51)
    LON, LAT = np.meshgrid(lon, lat)
    field = (
        sst
        + 1.4*np.sin((LON-65)/8)
        - 1.1*np.cos((LAT-17)/5)
        + ssh*2.0
        + 0.8*np.sin((LON+LAT)/9)
    )
    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=field,
        colorbar=dict(title="°C"),
        hovertemplate="Lon %{x:.2f}°<br>Lat %{y:.2f}°<br>Value %{z:.2f}°C<extra></extra>",
    ))
    fig.update_layout(height=520, xaxis_title="Longitude", yaxis_title="Latitude")
    st.plotly_chart(fig, use_container_width=True)

    depth_choice = st.select_slider("View reconstructed depth", options=DEPTHS, value=100)
    depth_temp = float(np.interp(depth_choice, DEPTHS, profile))
    depth_field = field - (sst-depth_temp) * (0.55 + 0.35*np.exp(-depth_choice/250))
    fig2 = go.Figure(go.Heatmap(
        x=lon, y=lat, z=depth_field,
        colorbar=dict(title="°C"),
    ))
    fig2.update_layout(
        height=520,
        title=f"Reconstructed temperature at {depth_choice} m",
        xaxis_title="Longitude", yaxis_title="Latitude",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.subheader("Learned feature importance / attention proxy")
    imp = importance.sort_values("Importance", ascending=True)
    fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h"))
    fig.update_layout(height=430, xaxis_title="Relative contribution", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧩 Embedding design")
        st.markdown("""
        <div class="card">
        <span class="badge">CNN</span>
        <span class="badge">CBAM</span>
        <span class="badge">Transformer</span>
        <span class="badge">Depth embedding</span>
        <span class="badge">Multitask</span>
        <br><br>
        Surface variables are fused into a compact <b>Ocean Embedding Z</b>.
        The prototype then conditions Z on depth to reconstruct T(z).
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("### 🔍 Interpretation")
        st.write(
            "The displayed importance is a prototype attention/explainability proxy. "
            "In the production model it should be replaced by learned CBAM/Transformer "
            "attention maps and SHAP values calculated from real satellite inputs."
        )

with tabs[3]:
    st.subheader("Physics-aware diagnostics")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Surface consistency", f"{physics['surface_consistency']:.1f}%")
    pc2.metric("Vertical smoothness", f"{physics['smoothness']:.1f}%")
    pc3.metric("Gradient realism", f"{physics['gradient_realism']:.1f}%")

    grad = np.gradient(profile, DEPTHS)
    fig = go.Figure(go.Scatter(
        x=grad, y=DEPTHS, mode="lines+markers", name="dT/dz"
    ))
    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    fig.update_xaxes(title="Vertical temperature gradient (°C/m)")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Production physics-aware training should minimize temperature error plus "
        "surface-consistency, vertical-gradient, smoothness, and thermocline losses. "
        f"Current demo extreme-event weight: {extreme_weight:.1f}×."
    )

with tabs[4]:
    st.subheader("Independent ARGO-style validation panel")
    argo = synthetic_argo_validation(profile, uncertainty=unc, seed=42)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("RMSE", f"{argo['rmse']:.2f} °C")
    v2.metric("MAE", f"{argo['mae']:.2f} °C")
    v3.metric("R²", f"{argo['r2']:.4f}")
    v4.metric("Bias", f"{argo['bias']:+.2f} °C")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=argo["observed"], y=argo["predicted"],
        mode="markers", name="ARGO comparisons"
    ))
    mn = min(argo["observed"].min(), argo["predicted"].min())
    mx = max(argo["observed"].max(), argo["predicted"].max())
    fig.add_trace(go.Scatter(
        x=[mn,mx], y=[mn,mx], mode="lines", name="1:1 line"
    ))
    fig.update_layout(height=480, xaxis_title="ARGO reference (°C)",
                      yaxis_title="OceanEmbed prediction (°C)")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "This panel is a functioning validation demo using generated ARGO-like reference "
        "samples. For SIH scientific results, replace it with actual independent INCOIS "
        "LAS Gridded ARGO observations and report spatial/seasonal/depth-wise metrics."
    )

with tabs[5]:
    st.subheader("OceanEmbed-NIO prototype specification")
    st.markdown("""
    **Core pipeline**

    `SST + SSS + SSH/SLA + U/V currents + U/V winds`
    → **QC & daily 0.25° harmonization**
    → **CNN spatial encoder**
    → **CBAM channel/spatial attention**
    → **Temporal Transformer**
    → **Ocean Embedding Z**
    → **Depth embedding + decoder**
    → **15-depth temperature profile**

    **Auxiliary outputs:** thermocline depth, OHC, uncertainty, explainability.

    **Scientific training plan:** GLORYS temperature as the primary reconstruction target;
    independent INCOIS/Gridded ARGO for validation; regional evaluation over Arabian Sea,
    Bay of Bengal and equatorial Indian Ocean.

    **Prototype note:** the current app intentionally runs without multi-gigabyte ocean
    datasets or GPU requirements. The supplied code includes a clean integration point for
    replacing the emulator with a trained PyTorch model and real GLORYS/ARGO data.
    """)

    metrics = pd.DataFrame({
        "Component": [
            "Spatial encoder", "Attention", "Temporal model", "Depth conditioning",
            "Physics-aware loss", "Uncertainty", "Independent validation",
            "Extreme-event weighting", "Explainability"
        ],
        "Prototype": [
            "Emulated", "Emulated", "Emulated", "Implemented",
            "Diagnostics implemented", "Implemented", "Synthetic demo",
            "Implemented", "Feature proxy"
        ],
        "Production upgrade": [
            "PyTorch CNN", "CBAM", "Transformer", "Learned depth embeddings",
            "Train with GLORYS", "MC Dropout / ensemble",
            "Real ARGO", "Event-weighted batches", "Attention + SHAP"
        ]
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown("### Export current prediction")
    export_df = pd.DataFrame({
        "date": [str(date)] * len(DEPTHS),
        "region": [region_name] * len(DEPTHS),
        "depth_m": DEPTHS,
        "temperature_c": np.round(profile, 4),
        "uncertainty_c": np.round(unc, 4),
    })
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download prediction CSV",
        data=csv,
        file_name=f"oceanembed_{date}_{region_name.replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    config = {
        "date": str(date), "region": region_name, "temporal_window_days": window,
        "attention_strength": attention_strength,
        "mc_dropout_runs": uncertainty_runs,
        "extreme_event_weight": extreme_weight,
        "surface_inputs": {
            "SST_C": sst, "SSS_PSU": sss, "SSH_anomaly_m": ssh,
            "U_current_mps": u_cur, "V_current_mps": v_cur,
            "U_wind_mps": u_wind, "V_wind_mps": v_wind,
        }
    }
    st.download_button(
        "⬇️ Download experiment configuration",
        data=json.dumps(config, indent=2),
        file_name="oceanembed_experiment.json",
        mime="application/json",
        use_container_width=True,
    )

st.markdown("---")
st.caption(
    "OceanEmbed-NIO • SIH26066 prototype • Research/demo software, not an operational "
    "oceanographic product. Predictions shown without real GLORYS/ARGO inputs are illustrative."
)
