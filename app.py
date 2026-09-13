import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oceanembed import (
    DEPTHS, REGIONS, generate_surface_state, reconstruct_profile,
    estimate_thermocline, compute_ohc, uncertainty_estimate,
    physics_diagnostics, synthetic_argo_validation, feature_importance,
)

st.set_page_config(page_title="OceanEmbed-NIO", page_icon="🌊", layout="wide")

# ---------------- State ----------------
for k, v in {
    "sst": 28.0, "sss": 35.0, "ssh": 0.05,
    "u_cur": 0.10, "v_cur": 0.05, "u_wind": 3.0, "v_wind": 1.0,
}.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Custom": None,
    "☀️ Calm ocean": dict(sst=28.0, sss=35.0, ssh=0.02, u_cur=0.08, v_cur=0.03, u_wind=2.0, v_wind=0.8),
    "🔥 Marine heatwave": dict(sst=32.0, sss=34.7, ssh=0.25, u_cur=0.35, v_cur=0.20, u_wind=2.5, v_wind=1.0),
    "🌪️ Strong mixing": dict(sst=29.0, sss=35.2, ssh=-0.12, u_cur=0.45, v_cur=-0.35, u_wind=12.0, v_wind=7.0),
    "🌊 Cyclone-like mixing": dict(sst=30.0, sss=35.1, ssh=-0.30, u_cur=0.60, v_cur=-0.55, u_wind=-15.0, v_wind=12.0),
}

st.markdown("""
<style>
.main { background: #0b0f14; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.hero { padding: 1.3rem 1.5rem; border-radius: 20px; background: linear-gradient(135deg, #063970, #0b7285); color: white; margin-bottom: 1rem; }
.hero h1 { margin: 0 0 .3rem 0; }
.hero p { margin: .2rem 0; opacity: .93; }
.card { background: #ffffff !important; border: 1px solid #e6edf3 !important; border-radius: 14px !important; padding: 0.8rem !important; color: #111111 !important; }
h1, h2, h3 { color: white; }
p, label { color: #eeeeee; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, icon=""):
    st.html(f"""
    <div style="width:100%;height:82px;box-sizing:border-box;background:#ffffff;border:1px solid #dfe6ed;border-radius:14px;padding:10px 12px;margin:0;overflow:hidden;font-family:Arial,sans-serif;">
        <div style="color:#111111;font-size:13px;font-weight:600;line-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0 0 5px 0;">{icon} {label}</div>
        <div style="color:#111111;font-size:22px;font-weight:700;line-height:26px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0;">{value}</div>
    </div>
    """)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Experiment controls")
region_name = st.sidebar.selectbox("Ocean region", list(REGIONS.keys()), index=0)
date = st.sidebar.date_input("Analysis date", value=pd.Timestamp("2026-09-12").date())
preset = st.sidebar.selectbox("Scenario preset", list(PRESETS.keys()))

if preset != "Custom" and st.sidebar.button("Apply scenario", use_container_width=True):
    for k, v in PRESETS[preset].items(): st.session_state[k] = v
    st.rerun()

st.sidebar.subheader("🛰️ Surface observations")
sst = st.sidebar.slider("SST (°C)",18.0,34.0,step=0.1,key="sst")
sss = st.sidebar.slider("SSS (PSU)",30.0,38.0,step=0.1,key="sss")
ssh = st.sidebar.slider("SSH / SLA (m)",-0.8,0.8,step=0.01,key="ssh")
u_cur = st.sidebar.slider("U current (m/s)",-1.5,1.5,step=0.01,key="u_cur")
v_cur = st.sidebar.slider("V current (m/s)",-1.5,1.5,step=0.01,key="v_cur")
u_wind = st.sidebar.slider("U wind (m/s)",-20.0,20.0,step=0.1,key="u_wind")
v_wind = st.sidebar.slider("V wind (m/s)",-20.0,20.0,step=0.1,key="v_wind")

st.sidebar.subheader("🧠 Model controls")
window = st.sidebar.select_slider("Temporal window (days)",[3,5,7,14],value=7)
attention = st.sidebar.slider("Attention strength",0.2,1.0,0.75,0.05)
extreme = st.sidebar.slider("Extreme-event weight",1.0,5.0,2.0,0.1)
runs = st.sidebar.slider("Uncertainty runs",10,60,30,5)
show_unc = st.sidebar.checkbox("Show uncertainty",True)

c1,c2=st.sidebar.columns(2)
if c1.button("🎲 Random",use_container_width=True):
    rng=np.random.default_rng()
    for k,lo,hi,dec in [("sst",25,32,1),("sss",33.5,36.5,1),("ssh",-.4,.4,2),("u_cur",-.8,.8,2),("v_cur",-.8,.8,2),("u_wind",-15,15,1),("v_wind",-15,15,1)]:
        st.session_state[k]=round(float(rng.uniform(lo,hi)),dec)
    st.rerun()
if c2.button("🔄 Reset",use_container_width=True):
    for k,v in dict(sst=28.0,sss=35.0,ssh=.05,u_cur=.1,v_cur=.05,u_wind=3.0,v_wind=1.0).items(): st.session_state[k]=v
    st.rerun()

# IMPORTANT: region is the string key, not REGIONS[region_name].
region = region_name
surface = generate_surface_state(region,pd.Timestamp(date),sst,sss,ssh,u_cur,v_cur,u_wind,v_wind,window)
profile = reconstruct_profile(surface,attention,extreme)
thermocline = estimate_thermocline(DEPTHS,profile)
ohcc = compute_ohc(DEPTHS,profile)
unc = uncertainty_estimate(surface,attention,extreme,runs)
physics = physics_diagnostics(DEPTHS,profile,sst)
importance = feature_importance(surface)
confidence=max(50,min(99,100-unc.mean()*35))

# ---------------- KPI strip ----------------
a,b,c,d,e=st.columns(5)
with a: metric_card("Thermocline", f"{thermocline:.1f} m", "🌡️")
with b: metric_card("OHC proxy", f"{ohcc:,.0f} MJ/m²", "🔥")
with c: metric_card("Uncertainty", f"±{unc.mean():.2f} °C", "📏")
with d: metric_card("Physics", f"{physics['score']:.1f}/100", "⚛️")
with e: metric_card("Confidence", f"{confidence:.0f}%", "🎯")

st.caption("Interactive emulator/demo. Real GLORYS training and independent INCOIS/Gridded ARGO validation are required for scientific results.")

# ---------------- Tabs ----------------
t1,t2,t3,t4,t5,t6,t7 = st.tabs(["🌡️ Profile Explorer","🗺️ Ocean Map","🧠 Attention","⚛️ Physics","🎯 ARGO","🔬 Compare Regions","📦 Export"])

with t1:
    st.subheader("Explore the reconstructed profile")
    depth=st.select_slider("Choose depth",options=list(DEPTHS),value=100)
    temp=float(np.interp(depth,DEPTHS,profile)); sigma=float(np.interp(depth,DEPTHS,unc))
    x,y,z=st.columns(3)
    with x: metric_card("Depth", f"{depth:.0f} m")
    with y: metric_card("Temperature", f"{temp:.2f} °C")
    with z: metric_card("Uncertainty", f"±{sigma:.2f} °C")
    fig=go.Figure()
    if show_unc:
        fig.add_trace(go.Scatter(x=profile+unc,y=DEPTHS,line=dict(width=0),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=profile-unc,y=DEPTHS,fill="tonexty",fillcolor="rgba(30,136,229,.12)",line=dict(width=0),name="Uncertainty"))
    fig.add_trace(go.Scatter(x=profile,y=DEPTHS,mode="lines+markers",name="OceanEmbed",line=dict(width=4),customdata=np.column_stack([unc]),hovertemplate="%{x:.2f} °C<br>%{y:.0f} m<br>±%{customdata[0]:.2f} °C<extra></extra>"))
    fig.add_hline(y=thermocline,line_dash="dash",annotation_text=f"Thermocline {thermocline:.0f}m")
    fig.add_hline(y=depth,line_dash="dot",annotation_text=f"Selected {depth:.0f}m")
    fig.update_yaxes(autorange="reversed",title="Depth (m)"); fig.update_xaxes(title="Temperature (°C)"); fig.update_layout(height=600)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({"Depth (m)":DEPTHS,"Temperature (°C)":np.round(profile,3),"Uncertainty (°C)":np.round(unc,3)}),use_container_width=True,hide_index=True)

with t2:
    st.subheader("Interactive North Indian Ocean map")
    depth=st.select_slider("Map depth",options=list(DEPTHS),value=100,key="map_depth")
    lon=np.linspace(45,105,61); lat=np.linspace(5,30,51); LON,LAT=np.meshgrid(lon,lat)
    field=sst+1.4*np.sin((LON-65)/8)-1.1*np.cos((LAT-17)/5)+ssh*2+0.8*np.sin((LON+LAT)/9)
    dt=float(np.interp(depth,DEPTHS,profile)); zfield=field-(sst-dt)*(0.55+0.35*np.exp(-depth/250))
    fig=go.Figure(go.Heatmap(x=lon,y=lat,z=zfield,colorscale="Turbo",colorbar=dict(title="°C"),hovertemplate="Lon %{x:.2f}°<br>Lat %{y:.2f}°<br>%{z:.2f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=[REGIONS[region_name]["lon"]],y=[REGIONS[region_name]["lat"]],mode="markers+text",text=[region_name],textposition="top center",marker=dict(size=15,symbol="star"),name="Selected region"))
    fig.update_layout(height=600,title=f"Reconstructed temperature at {depth:.0f} m",xaxis_title="Longitude",yaxis_title="Latitude")
    st.plotly_chart(fig,use_container_width=True)
    st.info("Map is an emulator visualization. Production OceanEmbed should render the learned daily 0.25° field.")

with t3:
    st.subheader("Feature contribution / attention proxy")
    imp=importance.sort_values("Importance")
    fig=go.Figure(go.Bar(x=imp.Importance,y=imp.Feature,orientation="h",text=(imp.Importance*100).round(1).astype(str)+"%",textposition="outside"))
    fig.update_layout(height=460,xaxis_title="Relative contribution")
    st.plotly_chart(fig,use_container_width=True)
    feat=st.selectbox("Inspect feature",list(importance.Feature))
    val=float(importance.loc[importance.Feature==feat,"Importance"].iloc[0])
    metric_card(f"{feat} contribution", f"{val*100:.1f}%", "🧠")
    st.markdown('<div class="card"><b>Production:</b> replace this proxy with learned CBAM maps, Transformer attention and SHAP on real satellite data.</div>',unsafe_allow_html=True)

with t4:
    st.subheader("Physics-aware diagnostics")
    p1,p2,p3,p4=st.columns(4)
    with p1: metric_card("Surface consistency", f"{physics['surface_consistency']:.1f}%")
    with p2: metric_card("Smoothness", f"{physics['smoothness']:.1f}%")
    with p3: metric_card("Gradient realism", f"{physics['gradient_realism']:.1f}%")
    with p4: metric_card("Overall", f"{physics['score']:.1f}/100")
    grad=np.gradient(profile,DEPTHS)
    fig=go.Figure(go.Scatter(x=grad,y=DEPTHS,mode="lines+markers",name="dT/dz")); fig.add_hline(y=thermocline,line_dash="dash",annotation_text="Thermocline"); fig.update_yaxes(autorange="reversed",title="Depth (m)"); fig.update_xaxes(title="dT/dz (°C/m)"); fig.update_layout(height=500)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({"Depth":DEPTHS,"Temp":profile,"dT/dz":grad}).round(5),use_container_width=True,hide_index=True)

with t5:
    st.subheader("ARGO-style validation")
    argo=synthetic_argo_validation(profile,uncertainty=unc,seed=42)
    a,b,c,d=st.columns(4)
    with a: metric_card("RMSE", f"{argo['rmse']:.2f} °C")
    with b: metric_card("MAE", f"{argo['mae']:.2f} °C")
    with c: metric_card("R²", f"{argo['r2']:.4f}")
    with d: metric_card("Bias", f"{argo['bias']:+.2f} °C")
    fig=go.Figure(go.Scatter(x=argo["observed"],y=argo["predicted"],mode="markers",customdata=DEPTHS,hovertemplate="Depth %{customdata:.0f}m<br>Ref %{x:.2f}°C<br>Pred %{y:.2f}°C<extra></extra>")); mn=min(argo["observed"].min(),argo["predicted"].min()); mx=max(argo["observed"].max(),argo["predicted"].max()); fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",name="1:1")); fig.update_layout(height=500,xaxis_title="Reference",yaxis_title="Prediction"); st.plotly_chart(fig,use_container_width=True)
    st.warning("Synthetic validation only. Replace with independent INCOIS LAS Gridded ARGO data for SIH evaluation.")

with t6:
    st.subheader("Compare regional profiles")
    selected=st.multiselect("Select regions",list(REGIONS.keys()),default=["Bay of Bengal","Arabian Sea"],max_selections=4)
    fig=go.Figure()
    for r in selected:
        s=generate_surface_state(r,pd.Timestamp(date),sst,sss,ssh,u_cur,v_cur,u_wind,v_wind,window)
        p=reconstruct_profile(s,attention,extreme); tc=estimate_thermocline(DEPTHS,p)
        fig.add_trace(go.Scatter(x=p,y=DEPTHS,mode="lines+markers",name=f"{r} • TC {tc:.0f}m"))
    fig.update_yaxes(autorange="reversed",title="Depth (m)"); fig.update_xaxes(title="Temperature (°C)"); fig.update_layout(height=600); st.plotly_chart(fig,use_container_width=True)

with t7:
    st.subheader("Download current experiment")
    df=pd.DataFrame({"date":[str(date)]*len(DEPTHS),"region":[region_name]*len(DEPTHS),"depth_m":DEPTHS,"temperature_c":profile,"uncertainty_c":unc})
    config={"date":str(date),"region":region_name,"window_days":window,"attention_strength":attention,"extreme_weight":extreme,"surface_inputs":{"SST_C":sst,"SSS_PSU":sss,"SSH_m":ssh,"U_current_mps":u_cur,"V_current_mps":v_cur,"U_wind_mps":u_wind,"V_wind_mps":v_wind},"outputs":{"thermocline_m":thermocline,"ohc_proxy_MJ_m2":ohcc,"mean_uncertainty_C":float(unc.mean()),"physics_score":physics["score"],"confidence_percent":confidence}}
    x,y=st.columns(2)
    x.download_button("⬇️ Temperature CSV",df.to_csv(index=False),file_name="oceanembed_prediction.csv",mime="text/csv",use_container_width=True)
    y.download_button("⬇️ Experiment JSON",json.dumps(config,indent=2),file_name="oceanembed_experiment.json",mime="application/json",use_container_width=True)
    st.dataframe(df.round(4),use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("OceanEmbed-NIO • SIH26066 • Interactive research/demo software; current predictions are generated by the included emulator.")
