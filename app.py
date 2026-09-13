import json
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from oceanembed import (
    DEPTHS, REGIONS, generate_surface_state, reconstruct_profile,
    estimate_thermocline, compute_ohc, uncertainty_estimate,
    physics_diagnostics, synthetic_argo_validation, feature_importance,
)

st.set_page_config(
    page_title="OceanEmbed NIO | Mission Control",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# STATE
# ----------------------------
DEFAULTS = dict(sst=28.0, sss=35.0, ssh=0.05, u_cur=0.10, v_cur=0.05, u_wind=3.0, v_wind=1.0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "Custom": None,
    "☀️ Calm ocean": dict(sst=28.0, sss=35.0, ssh=0.02, u_cur=0.08, v_cur=0.03, u_wind=2.0, v_wind=0.8),
    "🔥 Marine heatwave": dict(sst=32.0, sss=34.7, ssh=0.25, u_cur=0.35, v_cur=0.20, u_wind=2.5, v_wind=1.0),
    "🌪️ Strong mixing": dict(sst=29.0, sss=35.2, ssh=-0.12, u_cur=0.45, v_cur=-0.35, u_wind=12.0, v_wind=7.0),
    "🌀 Cyclone-like": dict(sst=30.0, sss=35.1, ssh=-0.30, u_cur=0.60, v_cur=-0.55, u_wind=-15.0, v_wind=12.0),
}

# ----------------------------
# VISUAL SYSTEM
# ----------------------------
st.markdown("""
<style>
:root{--bg:#050b14;--panel:#0b1422;--panel2:#0f1c2d;--line:#1c3146;--cyan:#39d9ff;--blue:#548dff;--green:#42e5a5;--amber:#ffd166;--red:#ff637d;--text:#eef8ff;--muted:#8199ad}
.stApp{background:radial-gradient(circle at 70% 0%,#0c2034 0,#050b14 38%,#040912 100%);color:var(--text)}
[data-testid="stHeader"]{background:rgba(5,11,20,.92)}
[data-testid="stSidebar"]{background:#070f1a;border-right:1px solid var(--line)}
.block-container{max-width:1580px;padding-top:1rem;padding-bottom:3rem}
h1,h2,h3,h4{color:var(--text)!important}
label,[data-testid="stWidgetLabel"] p{color:#cfe2f0!important}
.stMarkdown p{color:#a7bdce}

.topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 2px 18px;border-bottom:1px solid #17293b;margin-bottom:18px}
.brand{font-size:1.05rem;font-weight:850;letter-spacing:.08em;color:#e9f8ff}.brand span{color:var(--cyan)}
.live{font-size:.72rem;color:#75f1b9;border:1px solid #235b4b;background:#092219;padding:6px 10px;border-radius:999px;font-weight:800;letter-spacing:.08em}

.hero{border:1px solid #1d4e69;border-radius:24px;padding:22px 24px;background:linear-gradient(135deg,rgba(17,55,78,.72),rgba(7,17,29,.96) 55%,rgba(13,31,50,.92));box-shadow:0 24px 70px rgba(0,0,0,.25);position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;width:360px;height:360px;right:-130px;top:-180px;border-radius:50%;border:1px solid rgba(57,217,255,.18);box-shadow:0 0 0 30px rgba(57,217,255,.03),0 0 0 60px rgba(57,217,255,.02)}
.hero-kicker{font-size:.72rem;color:#72e7ff;letter-spacing:.14em;font-weight:850}.hero h1{font-size:2.25rem;margin:5px 0 2px}.hero p{max-width:900px;color:#9ab4c8;margin:0}

.ribbon{display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:10px;margin:14px 0}.ribbon-item{background:#091321;border:1px solid var(--line);border-radius:14px;padding:10px 13px}.ribbon-label{font-size:.66rem;color:#718ca2;text-transform:uppercase;letter-spacing:.1em}.ribbon-value{font-size:.98rem;color:#eaf7ff;font-weight:800;margin-top:3px}

.section{font-size:1.02rem;font-weight:850;color:#e5f6ff;margin:22px 0 9px;display:flex;align-items:center;gap:8px}.section:before{content:"";width:4px;height:18px;border-radius:4px;background:linear-gradient(#39d9ff,#548dff)}
.card{background:linear-gradient(145deg,#0d1928,#09121e);border:1px solid var(--line);border-radius:18px;padding:15px;min-height:112px;box-shadow:0 12px 34px rgba(0,0,0,.16)}
.card .ico{font-size:1.15rem}.card .lab{font-size:.69rem;color:#7891a6;text-transform:uppercase;letter-spacing:.08em;margin-top:8px}.card .val{font-size:1.5rem;font-weight:900;color:#f5fbff;line-height:1.2;margin-top:3px}.card .sub{font-size:.69rem;color:#70899d;margin-top:4px}

.panel{background:#081321;border:1px solid var(--line);border-radius:20px;padding:14px 16px;box-shadow:0 14px 40px rgba(0,0,0,.14)}.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:9px}.panel-title{font-size:.96rem;font-weight:850;color:#e8f6ff}.panel-note{font-size:.7rem;color:#71899c}.tag{font-size:.65rem;padding:4px 8px;border-radius:999px;border:1px solid #24445b;color:#88a9bf;background:#0c1b2b}

.alert{padding:11px 13px;border-radius:13px;border:1px solid;margin:8px 0;font-size:.78rem;font-weight:750}.green{background:#08251d;border-color:#1d654e;color:#6ff0b8}.amber{background:#2a2110;border-color:#66501e;color:#ffd873}.red{background:#2b121a;border-color:#6c2839;color:#ff91a5}.blue{background:#092133;border-color:#1c5270;color:#77e7ff}

[data-testid="stTabs"]{margin-top:8px}.stTabs [data-baseweb="tab-list"]{gap:5px;background:#07101b;border:1px solid var(--line);padding:6px;border-radius:15px}.stTabs [data-baseweb="tab"]{border-radius:10px;color:#829aae;font-weight:750;padding:9px 14px}.stTabs [aria-selected="true"]{background:#10283a;color:#67e5ff!important}
.stButton>button,.stDownloadButton>button{border-radius:10px!important;border:1px solid #29465d!important;background:#0d2032!important;color:#e7f7ff!important;font-weight:750!important}.stButton>button:hover{border-color:#39d9ff!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HELPERS
# ----------------------------
def card(title, value, icon, sub=""):
    st.markdown(f'''<div class="card"><div class="ico">{icon}</div><div class="lab">{title}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>''', unsafe_allow_html=True)

def panel(title, note="", tag=""):
    tag_html=f'<span class="tag">{tag}</span>' if tag else ''
    st.markdown(f'''<div class="panel"><div class="panel-head"><div><div class="panel-title">{title}</div><div class="panel-note">{note}</div></div>{tag_html}</div>''', unsafe_allow_html=True)

def end_panel(): st.markdown('</div>', unsafe_allow_html=True)

def alert(text, kind="blue"): st.markdown(f'<div class="alert {kind}">{text}</div>', unsafe_allow_html=True)

def fig_style(fig, height=420):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#07111c",font=dict(color="#dceef8"),height=height,margin=dict(l=35,r=25,t=45,b=35),legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#183044",zerolinecolor="#254359")
    fig.update_yaxes(gridcolor="#183044",zerolinecolor="#254359")
    return fig

def normalize(values):
    arr=np.asarray(values,dtype=float); lo,hi=arr.min(),arr.max()
    return np.zeros_like(arr) if hi==lo else (arr-lo)/(hi-lo)

def state_label(sst, physics_score, uncertainty):
    if sst>=31 or physics_score<60:return "HIGH ATTENTION","red",88
    if sst>=29.5 or uncertainty>.45 or physics_score<75:return "WATCH","amber",62
    return "STABLE","green",24

# ----------------------------
# SIDEBAR COMMAND CENTER
# ----------------------------
st.sidebar.markdown("# 🌊 OceanEmbed")
st.sidebar.caption("NIO Mission Control · SIH26066")
st.sidebar.markdown("---")
region_name=st.sidebar.selectbox("Ocean sector",list(REGIONS.keys()),index=0)
analysis_date=st.sidebar.date_input("Analysis date",value=date(2026,9,12))
preset=st.sidebar.selectbox("Operating scenario",list(PRESETS.keys()))
if preset!="Custom" and st.sidebar.button("Apply scenario",use_container_width=True):
    for k,v in PRESETS[preset].items():st.session_state[k]=v
    st.rerun()
with st.sidebar.expander("🛰️ Surface observation controls",expanded=True):
    sst=st.slider("SST (°C)",18.0,34.0,step=.1,key="sst")
    sss=st.slider("SSS (PSU)",30.0,38.0,step=.1,key="sss")
    ssh=st.slider("SSH / SLA (m)",-0.8,.8,step=.01,key="ssh")
    u_cur=st.slider("U current (m/s)",-1.5,1.5,step=.01,key="u_cur")
    v_cur=st.slider("V current (m/s)",-1.5,1.5,step=.01,key="v_cur")
    u_wind=st.slider("U wind (m/s)",-20.,20.,step=.1,key="u_wind")
    v_wind=st.slider("V wind (m/s)",-20.,20.,step=.1,key="v_wind")
with st.sidebar.expander("🧠 Model controls",expanded=False):
    window=st.select_slider("Temporal window",[3,5,7,14],value=7)
    attention=st.slider("Attention strength",.2,1.,.75,.05)
    extreme=st.slider("Extreme-event weight",1.,5.,2.,.1)
    runs=st.slider("Uncertainty ensemble runs",10,60,30,5)
    show_unc=st.checkbox("Display uncertainty",True)
with st.sidebar.expander("🧭 Display controls",expanded=False):
    theme_mode=st.selectbox("Chart density",["Operational","Research","Presentation"])
    show_markers=st.checkbox("Map region markers",True)
    show_grid=st.checkbox("Show gridlines",True)
r1,r2=st.sidebar.columns(2)
if r1.button("🎲 Random",use_container_width=True):
    rng=np.random.default_rng()
    for k,lo,hi,d in [("sst",25,32,1),("sss",33.5,36.5,1),("ssh",-.4,.4,2),("u_cur",-.8,.8,2),("v_cur",-.8,.8,2),("u_wind",-15,15,1),("v_wind",-15,15,1)]:st.session_state[k]=round(float(rng.uniform(lo,hi)),d)
    st.rerun()
if r2.button("↺ Reset",use_container_width=True):
    for k,v in DEFAULTS.items():st.session_state[k]=v
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption("Prototype / emulator mode")
st.sidebar.caption("Production path: satellite harmonization → GLORYS training → independent INCOIS/Gridded ARGO validation.")

# ----------------------------
# MODEL
# ----------------------------
surface=generate_surface_state(region_name,pd.Timestamp(analysis_date),sst,sss,ssh,u_cur,v_cur,u_wind,v_wind,window)
profile=reconstruct_profile(surface,attention,extreme)
thermocline=estimate_thermocline(DEPTHS,profile)
ohcc=compute_ohc(DEPTHS,profile)
unc=uncertainty_estimate(surface,attention,extreme,runs)
physics=physics_diagnostics(DEPTHS,profile,sst)
importance=feature_importance(surface)
mean_unc=float(np.mean(unc)); confidence=max(50,min(99,100-mean_unc*35))
gradient=np.gradient(profile,DEPTHS)
current_speed=float(np.hypot(u_cur,v_cur)); wind_speed=float(np.hypot(u_wind,v_wind))
mixing_index=float(np.clip(45+wind_speed*2.7+current_speed*22+abs(ssh)*35,0,100))
stratification=float(np.clip(100-np.mean(np.abs(gradient))*900,0,100))
anomaly_index=float(np.clip((sst-28)*14+ssh*35,-100,100))
risk,risk_kind,risk_value=state_label(sst,physics["score"],mean_unc)

# ----------------------------
# HEADER
# ----------------------------
st.markdown('''<div class="topbar"><div class="brand">OCEAN<span>EMBED</span> / NIO MISSION CONTROL</div><div class="live">● EMULATOR ONLINE</div></div>''',unsafe_allow_html=True)
st.markdown(f'''<div class="hero"><div class="hero-kicker">SIH26066 · DAILY 0.25° CONCEPT · {region_name.upper()}</div><h1>Subsurface Ocean Intelligence</h1><p>Explore surface forcing, reconstructed vertical thermal structure, spatial fields, model attention, physics consistency, uncertainty and validation readiness from one operational-style console.</p></div>''',unsafe_allow_html=True)

st.markdown(f'''<div class="ribbon"><div class="ribbon-item"><div class="ribbon-label">Current state</div><div class="ribbon-value">● {risk} · {region_name}</div></div><div class="ribbon-item"><div class="ribbon-label">Analysis</div><div class="ribbon-value">{analysis_date.strftime('%d %b %Y')}</div></div><div class="ribbon-item"><div class="ribbon-label">Scenario</div><div class="ribbon-value">{preset}</div></div><div class="ribbon-item"><div class="ribbon-label">Confidence</div><div class="ribbon-value">{confidence:.0f}% · ±{mean_unc:.2f} °C</div></div></div>''',unsafe_allow_html=True)

# KPI COMMAND DECK
st.markdown('<div class="section">Mission telemetry</div>',unsafe_allow_html=True)
a,b,c,d,e=st.columns(5,gap="medium")
with a: card("Thermocline","%.1f m"%thermocline,"🌡️","vertical transition depth")
with b: card("OHC proxy","%.0f MJ/m²"%ohcc,"🔥","upper-ocean heat")
with c: card("Physics score","%.1f/100"%physics["score"],"⚛️","consistency index")
with d: card("Mixing index","%.0f/100"%mixing_index,"🌪️","forcing diagnostic")
with e: card("Risk state",risk,"🚦","attention index %d/100"%risk_value)

# MAIN NAV
T=st.tabs(["🛰️ Command Center","🌡️ 3D Profile Lab","🗺️ Spatial Digital Twin","🧠 AI Studio","⚛️ Physics Monitor","🎯 Validation Hub","🧪 Scenario Simulator","📦 Data & Research"])

# =========================
# COMMAND CENTER
# =========================
with T[0]:
    st.markdown('<div class="section">Operational overview</div>',unsafe_allow_html=True)
    left,mid,right=st.columns([1.55,1.05,0.85],gap="medium")
    with left:
        panel("Vertical thermal structure","15 target depths · hover for uncertainty","CORE MODEL")
        fig=go.Figure()
        if show_unc:
            fig.add_trace(go.Scatter(x=profile+unc,y=DEPTHS,line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=profile-unc,y=DEPTHS,fill="tonexty",fillcolor="rgba(57,217,255,.13)",line=dict(width=0),name="± uncertainty"))
        fig.add_trace(go.Scatter(x=profile,y=DEPTHS,mode="lines+markers",name="Reconstruction",line=dict(color="#55dcff",width=4),marker=dict(size=7),customdata=unc,hovertemplate="%{x:.2f} °C · %{y:.0f} m · ±%{customdata:.2f} °C<extra></extra>"))
        fig.add_hline(y=thermocline,line_dash="dot",line_color="#ffd166",annotation_text="Thermocline")
        fig.update_yaxes(autorange="reversed",title="Depth (m)");fig.update_xaxes(title="Temperature (°C)")
        fig_style(fig,500);st.plotly_chart(fig,use_container_width=True)
        end_panel()
    with mid:
        panel("Surface forcing","Normalized model inputs","INPUT VECTOR")
        fdf=pd.DataFrame({"Feature":["SST","SSS","SSH","U current","V current","U wind","V wind"],"Value":[sst,sss,ssh,u_cur,v_cur,u_wind,v_wind]})
        fig=go.Figure(go.Bar(x=normalize(fdf.Value),y=fdf.Feature,orientation="h",marker_color="#548dff",text=fdf.Value.round(2),textposition="outside"))
        fig.update_xaxes(range=[0,1.18],title="relative magnitude");fig_style(fig,380)
        st.plotly_chart(fig,use_container_width=True)
        alert("High surface forcing detected." if wind_speed>12 or current_speed>.9 else "Surface forcing within emulator operating range.","amber" if wind_speed>12 or current_speed>.9 else "green")
        end_panel()
    with right:
        panel("Mission health","Fast-look operational indicators","HEALTH")
        for label,val,kind in [("Confidence",confidence,"green" if confidence>=80 else "amber"),("Physics",physics["score"],"green" if physics["score"]>=80 else "amber"),("Stability",stratification,"green" if stratification>=70 else "amber"),("Uncertainty",max(0,100-mean_unc*100),"green" if mean_unc<.35 else "amber")]:
            color={"green":"#42e5a5","amber":"#ffd166","red":"#ff637d"}[kind]
            st.markdown(f'''<div style="margin:13px 0"><div style="display:flex;justify-content:space-between;font-size:.72rem;color:#91a8ba"><span>{label}</span><b style="color:{color}">{val:.0f}</b></div><div style="height:7px;background:#142435;border-radius:99px;margin-top:5px"><div style="width:{min(100,max(0,val)):.1f}%;height:100%;background:{color};border-radius:99px"></div></div></div>''',unsafe_allow_html=True)
        alert(f"{risk} · attention index {risk_value}/100",risk_kind)
        st.caption("Operational-style emulator status; not an observed alert product.")
        end_panel()

    st.markdown('<div class="section">Deep-dive telemetry</div>',unsafe_allow_html=True)
    q1,q2,q3,q4=st.columns(4,gap="medium")
    with q1: card("SST","%.1f °C"%sst,"☀️","surface thermal state")
    with q2: card("Wind speed","%.1f m/s"%wind_speed,"💨","vector magnitude")
    with q3: card("Current speed","%.2f m/s"%current_speed,"🌊","surface current")
    with q4: card("Stratification","%.0f/100"%stratification,"📐","emulator proxy")

    st.markdown('<div class="section">Data lineage & readiness</div>',unsafe_allow_html=True)
    lineage=pd.DataFrame([
        ["01","Satellite inputs","SST · SSS · SSH/SLA · U/V currents · U/V winds","READY"],
        ["02","Harmonization","Daily 0.25° NIO grid + QC + missing mask","DESIGN"],
        ["03","Ocean embedding","CNN / attention / temporal representation","PROTOTYPE"],
        ["04","Depth decoder","15-level temperature profile","PROTOTYPE"],
        ["05","GLORYS target","Training reference at depth","REQUIRED"],
        ["06","Gridded ARGO","Independent validation","REQUIRED"],
    ],columns=["ID","Layer","Role","Status"])
    st.dataframe(lineage,use_container_width=True,hide_index=True)

# =========================
# PROFILE LAB
# =========================
with T[1]:
    st.markdown('<div class="section">Vertical profile laboratory</div>',unsafe_allow_html=True)
    top1,top2,top3,top4=st.columns(4,gap="medium")
    selected_depth=top1.select_slider("Inspection depth",options=list(DEPTHS),value=100,key="inspect_depth")
    temp=float(np.interp(selected_depth,DEPTHS,profile));sigma=float(np.interp(selected_depth,DEPTHS,unc))
    with top2: card("Selected depth","%.0f m"%selected_depth,"📍","profile cursor")
    with top3: card("Temperature","%.2f °C"%temp,"🌡️","reconstructed")
    with top4: card("Uncertainty","±%.2f °C"%sigma,"📏","local estimate")
    pleft,pright=st.columns([1.45,.85],gap="medium")
    with pleft:
        panel("Profile explorer","Select a depth above and inspect the complete reconstruction","INTERACTIVE")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=profile,y=DEPTHS,mode="lines+markers",name="Temperature",line=dict(color="#ff91a8",width=4)))
        if show_unc:
            fig.add_trace(go.Scatter(x=profile+unc,y=DEPTHS,line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=profile-unc,y=DEPTHS,fill="tonexty",fillcolor="rgba(84,141,255,.12)",line=dict(width=0),name="Uncertainty"))
        fig.add_trace(go.Scatter(x=[temp],y=[selected_depth],mode="markers",marker=dict(size=16,color="#ffd166",line=dict(width=2,color="#fff")),name="Cursor"))
        fig.add_hline(y=thermocline,line_dash="dot",line_color="#ffd166",annotation_text=f"Thermocline {thermocline:.0f} m")
        fig.update_yaxes(autorange="reversed",title="Depth (m)");fig.update_xaxes(title="Temperature (°C)")
        fig_style(fig,650);st.plotly_chart(fig,use_container_width=True)
        end_panel()
    with pright:
        panel("Vertical gradient","Thermal change with depth","dT/dz")
        gf=go.Figure(go.Bar(x=gradient,y=DEPTHS,orientation="h",marker_color=np.where(gradient<0,"#39d9ff","#ff637d")))
        gf.add_vline(x=0,line_color="#627789");gf.update_yaxes(autorange="reversed",title="Depth (m)");gf.update_xaxes(title="°C/m")
        fig_style(gf,420);st.plotly_chart(gf,use_container_width=True)
        panel_df=pd.DataFrame({"Depth":DEPTHS,"Temp °C":np.round(profile,3),"± °C":np.round(unc,3),"dT/dz":np.round(gradient,5)})
        st.dataframe(panel_df,use_container_width=True,hide_index=True,height=240)
        end_panel()

# =========================
# SPATIAL DIGITAL TWIN
# =========================
with T[2]:
    st.markdown('<div class="section">North Indian Ocean spatial digital twin</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns([.8,1,1,1],gap="medium")
    map_depth=c1.select_slider("Depth",options=list(DEPTHS),value=100,key="map_depth2")
    field_mode=c2.selectbox("Field",["Temperature","Uncertainty","Anomaly","Gradient proxy"])
    palette=c3.selectbox("Palette",["Thermal","Ocean","Diverging"])
    smooth=c4.slider("Spatial contrast",.5,2.,1.,.1)
    lon=np.linspace(45,105,121);lat=np.linspace(5,30,101);LON,LAT=np.meshgrid(lon,lat)
    dt=float(np.interp(map_depth,DEPTHS,profile))
    spatial=(sst+1.4*np.sin((LON-65)/8)-1.1*np.cos((LAT-17)/5)+ssh*2+.8*np.sin((LON+LAT)/9))-(sst-dt)*(0.55+.35*np.exp(-map_depth/250))
    uf=mean_unc*(.75+.4*np.abs(np.sin((LON-60)/12))*np.cos((LAT-17)/7)**2)
    af=spatial-np.mean(spatial)
    gf=np.gradient(spatial,axis=0)
    field=spatial if field_mode=="Temperature" else uf if field_mode=="Uncertainty" else af if field_mode=="Anomaly" else gf
    cs="Turbo" if palette=="Thermal" else "Viridis" if palette=="Ocean" else "RdBu_r"
    mapfig=go.Figure(go.Heatmap(x=lon,y=lat,z=field* smooth,colorscale=cs,colorbar=dict(title=field_mode),hovertemplate="Lon %{x:.2f}°<br>Lat %{y:.2f}°<br>%{z:.2f}<extra></extra>"))
    if show_markers:
        mapfig.add_trace(go.Scatter(x=[REGIONS[r]["lon"] for r in REGIONS],y=[REGIONS[r]["lat"] for r in REGIONS],text=list(REGIONS.keys()),mode="markers+text",textposition="top center",marker=dict(size=8,color="#fff",line=dict(color="#05101a",width=2)),name="Sectors"))
    mapfig.update_xaxes(title="Longitude (°E)");mapfig.update_yaxes(title="Latitude (°N)");fig_style(mapfig,680)
    st.plotly_chart(mapfig,use_container_width=True)
    b1,b2,b3=st.columns(3)
    with b1: card("Map depth","%.0f m"%map_depth,"🗺️","selected layer")
    with b2: card("Field mean","%.2f"%float(np.mean(field)),"∿","spatial statistic")
    with b3: card("Field spread","%.2f"%float(np.std(field)),"σ","spatial variability")
    alert("Spatial map is generated by the prototype emulator. Production deployment should render the learned 0.25° field from the trained network.","blue")

# =========================
# AI STUDIO
# =========================
with T[3]:
    st.markdown('<div class="section">AI embedding & explainability studio</div>',unsafe_allow_html=True)
    left,right=st.columns([1.25,.9],gap="medium")
    with left:
        panel("Feature contribution matrix","Relative importance of the surface predictors","ATTENTION")
        imp=importance.sort_values("Importance")
        f=go.Figure(go.Bar(x=imp.Importance,y=imp.Feature,orientation="h",marker_color="#39d9ff",text=(imp.Importance*100).round(1).astype(str)+"%",textposition="outside"))
        f.update_xaxes(range=[0,max(1,imp.Importance.max()*1.25)],title="relative contribution");fig_style(f,500);st.plotly_chart(f,use_container_width=True)
        end_panel()
    with right:
        panel("Embedding diagnostics","Conceptual latent-space telemetry","MODEL")
        latent=np.array([np.mean(profile),np.std(profile),thermocline,ohcc/100,mean_unc,physics["score"],mixing_index,stratification])
        labels=["Mean T","T spread","Thermocline","OHC","Uncertainty","Physics","Mixing","Stratification"]
        radar=go.Figure(go.Scatterpolar(r=normalize(latent),theta=labels,fill="toself",line=dict(color="#548dff",width=2),fillcolor="rgba(84,141,255,.16)"))
        radar.update_layout(polar=dict(bgcolor="#07111c",radialaxis=dict(visible=True,range=[0,1],gridcolor="#20384b"),angularaxis=dict(gridcolor="#20384b")))
        fig_style(radar,420);st.plotly_chart(radar,use_container_width=True)
        end_panel()
    st.markdown('<div class="section">Attention-by-depth view</div>',unsafe_allow_html=True)
    attention_profile=normalize(np.abs(gradient))*0.65+normalize(unc)*0.35
    h=go.Figure(go.Heatmap(z=np.array([attention_profile]),x=DEPTHS,y=["attention"],colorscale="Turbo",colorbar=dict(title="weight"),hovertemplate="Depth %{x:.0f} m<br>Attention %{z:.2f}<extra></extra>"))
    h.update_xaxes(title="Depth (m)");fig_style(h,230);st.plotly_chart(h,use_container_width=True)

# =========================
# PHYSICS MONITOR
# =========================
with T[4]:
    st.markdown('<div class="section">Physics consistency & risk monitor</div>',unsafe_allow_html=True)
    p1,p2,p3=st.columns([1.2,1,1],gap="medium")
    with p1:
        panel("Thermal gradient","Vertical structure diagnostic","PHYSICS")
        g=go.Figure(go.Scatter(x=gradient,y=DEPTHS,mode="lines+markers",line=dict(color="#ff8ca5",width=3)))
        g.add_vline(x=0,line_dash="dot",line_color="#70899d");g.update_yaxes(autorange="reversed",title="Depth (m)");g.update_xaxes(title="dT/dz (°C/m)");fig_style(g,430);st.plotly_chart(g,use_container_width=True);end_panel()
    with p2:
        panel("Attention index","Emulator-level operational indicator","RISK")
        gauge=go.Figure(go.Indicator(mode="gauge+number",value=risk_value,title={"text":"Attention"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#ff637d" if risk_kind=="red" else "#ffd166" if risk_kind=="amber" else "#42e5a5"},"steps":[{"range":[0,40],"color":"#0d2c24"},{"range":[40,70],"color":"#342b12"},{"range":[70,100],"color":"#381822"}]}))
        fig_style(gauge,350);st.plotly_chart(gauge,use_container_width=True);end_panel()
    with p3:
        panel("Diagnostic stack","Current experiment state","CHECKS")
        checks=[("Surface consistency",min(100,95-abs(sst-28)*4)),("Vertical smoothness",stratification),("Uncertainty quality",max(0,100-mean_unc*100)),("Model confidence",confidence)]
        for name,val in checks:
            st.markdown(f'<div style="margin:14px 0"><div style="display:flex;justify-content:space-between;color:#8ea7ba;font-size:.72rem"><span>{name}</span><b style="color:#eaf7ff">{val:.0f}%</b></div><div style="height:7px;background:#142435;border-radius:8px;margin-top:5px"><div style="width:{np.clip(val,0,100):.0f}%;height:100%;background:#39d9ff;border-radius:8px"></div></div></div>',unsafe_allow_html=True)
        end_panel()
    alert("High attention state" if risk_kind=="red" else "Watch state" if risk_kind=="amber" else "Stable state",risk_kind)

# =========================
# VALIDATION HUB
# =========================
with T[5]:
    st.markdown('<div class="section">Independent validation readiness</div>',unsafe_allow_html=True)
    argo=synthetic_argo_validation(profile,uncertainty=unc,seed=42)
    a,b,c,d=st.columns(4,gap="medium")
    with a: card("RMSE","%.2f °C"%argo["rmse"],"📉","synthetic reference")
    with b: card("MAE","%.2f °C"%argo["mae"],"📊","synthetic reference")
    with c: card("R²","%.4f"%argo["r2"],"🎯","synthetic reference")
    with d: card("Bias","%+.2f °C"%argo["bias"],"⚖️","mean signed error")
    left,right=st.columns([1.2,1],gap="medium")
    with left:
        panel("Prediction vs reference","Depth-wise validation scatter","VALIDATION")
        f=go.Figure(go.Scatter(x=argo["observed"],y=argo["predicted"],mode="markers",marker=dict(size=10,color="#39d9ff"),customdata=DEPTHS,hovertemplate="%{customdata:.0f} m<br>Ref %{x:.2f} °C<br>Pred %{y:.2f} °C<extra></extra>"))
        mn=min(argo["observed"].min(),argo["predicted"].min());mx=max(argo["observed"].max(),argo["predicted"].max())
        f.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(color="#ffd166",dash="dash"),name="1:1"));f.update_xaxes(title="Reference");f.update_yaxes(title="Prediction");fig_style(f,470);st.plotly_chart(f,use_container_width=True);end_panel()
    with right:
        panel("Residual depth profile","Prediction minus reference","ERROR")
        residual=argo["predicted"]-argo["observed"]
        rf=go.Figure(go.Bar(x=DEPTHS,y=residual,marker_color=np.where(residual>=0,"#ff7f99","#39d9ff")))
        rf.add_hline(y=0,line_dash="dot");rf.update_xaxes(title="Depth (m)");rf.update_yaxes(title="Residual (°C)");fig_style(rf,470);st.plotly_chart(rf,use_container_width=True);end_panel()
    alert("Synthetic validation only. Final SIH evidence should use independent INCOIS LAS Gridded ARGO observations.","amber")

# =========================
# SCENARIO SIMULATOR
# =========================
with T[6]:
    st.markdown('<div class="section">Scenario simulator & sensitivity lab</div>',unsafe_allow_html=True)
    selected=st.multiselect("Compare operating scenarios",list(PRESETS.keys())[1:],default=list(PRESETS.keys())[1:4],max_selections=4)
    records=[];fig=go.Figure()
    for name in selected:
        v=PRESETS[name];ss=generate_surface_state(region_name,pd.Timestamp(analysis_date),v["sst"],v["sss"],v["ssh"],v["u_cur"],v["v_cur"],v["u_wind"],v["v_wind"],window);pp=reconstruct_profile(ss,attention,extreme);tc=estimate_thermocline(DEPTHS,pp);oc=compute_ohc(DEPTHS,pp)
        records.append([name,v["sst"],tc,oc]);fig.add_trace(go.Scatter(x=pp,y=DEPTHS,mode="lines+markers",name=name))
    fig.update_yaxes(autorange="reversed",title="Depth (m)");fig.update_xaxes(title="Temperature (°C)");fig_style(fig,520);st.plotly_chart(fig,use_container_width=True)
    sdf=pd.DataFrame(records,columns=["Scenario","SST °C","Thermocline m","OHC MJ/m²"])
    if not sdf.empty: st.dataframe(sdf.round(2),use_container_width=True,hide_index=True)
    s1,s2=st.columns([1,1],gap="medium")
    with s1:
        sweep_feature=st.selectbox("Sensitivity variable",["SST","SSS","SSH/SLA","Wind speed"])
    with s2:
        sweep_points=st.slider("Sweep resolution",7,25,13)
    if sweep_feature=="SST": sweep=np.linspace(max(18,sst-3),min(34,sst+3),sweep_points)
    elif sweep_feature=="SSS": sweep=np.linspace(max(30,sss-1.5),min(38,sss+1.5),sweep_points)
    elif sweep_feature=="SSH/SLA": sweep=np.linspace(max(-.8,ssh-.35),min(.8,ssh+.35),sweep_points)
    else: sweep=np.linspace(max(0,wind_speed-8),min(28,wind_speed+8),sweep_points)
    outs=[]
    for val in sweep:
        vals=dict(sst=sst,sss=sss,ssh=ssh,u_cur=u_cur,v_cur=v_cur,u_wind=u_wind,v_wind=v_wind)
        if sweep_feature=="SST":vals["sst"]=float(val)
        elif sweep_feature=="SSS":vals["sss"]=float(val)
        elif sweep_feature=="SSH/SLA":vals["ssh"]=float(val)
        else:vals["u_wind"]=float(val);vals["v_wind"]=0
        ss=generate_surface_state(region_name,pd.Timestamp(analysis_date),vals["sst"],vals["sss"],vals["ssh"],vals["u_cur"],vals["v_cur"],vals["u_wind"],vals["v_wind"],window);pp=reconstruct_profile(ss,attention,extreme)
        outs.append([val,estimate_thermocline(DEPTHS,pp),compute_ohc(DEPTHS,pp)])
    sw=pd.DataFrame(outs,columns=["Input","Thermocline","OHC"])
    sf=go.Figure();sf.add_trace(go.Scatter(x=sw.Input,y=sw.Thermocline,mode="lines+markers",name="Thermocline"));sf.add_trace(go.Scatter(x=sw.Input,y=sw.OHC,mode="lines+markers",name="OHC",yaxis="y2"));sf.update_layout(yaxis=dict(title="Thermocline (m)"),yaxis2=dict(title="OHC (MJ/m²)",overlaying="y",side="right"));sf.update_xaxes(title=sweep_feature);fig_style(sf,450);st.plotly_chart(sf,use_container_width=True)

# =========================
# DATA & RESEARCH
# =========================
with T[7]:
    st.markdown('<div class="section">Data products, export & research traceability</div>',unsafe_allow_html=True)
    export_df=pd.DataFrame({"date":[str(analysis_date)]*len(DEPTHS),"region":[region_name]*len(DEPTHS),"depth_m":DEPTHS,"temperature_c":profile,"uncertainty_c":unc,"dT_dz_c_per_m":gradient})
    config={"date":str(analysis_date),"region":region_name,"scenario":preset,"window_days":window,"attention_strength":attention,"extreme_weight":extreme,"uncertainty_runs":runs,"surface_inputs":{"SST_C":sst,"SSS_PSU":sss,"SSH_m":ssh,"U_current_mps":u_cur,"V_current_mps":v_cur,"U_wind_mps":u_wind,"V_wind_mps":v_wind},"outputs":{"thermocline_m":thermocline,"ohc_proxy_MJ_m2":ohcc,"mean_uncertainty_C":mean_unc,"physics_score":physics["score"],"confidence_percent":confidence,"risk_state":risk},"prototype_note":"Emulator/demo output. Replace with GLORYS-trained model and independent INCOIS Gridded ARGO validation."}
    a,b,c=st.columns(3,gap="medium")
    with a: st.download_button("⬇ Temperature CSV",export_df.to_csv(index=False),"oceanembed_profile.csv","text/csv",use_container_width=True)
    with b: st.download_button("⬇ Experiment JSON",json.dumps(config,indent=2),"oceanembed_experiment.json","application/json",use_container_width=True)
    with c: st.download_button("⬇ Mission report TXT",("OCEANEMBED NIO MISSION REPORT\n\n"+json.dumps(config,indent=2)),"oceanembed_report.txt","text/plain",use_container_width=True)
    st.dataframe(export_df.round(4),use_container_width=True,hide_index=True)
    st.markdown('<div class="section">Implementation readiness matrix</div>',unsafe_allow_html=True)
    roadmap=pd.DataFrame([["Surface data","SST/SSS/SSH/U/V/wind","Prototype","GLORYS + satellite ingest required"],["0.25° daily grid","NIO 5–30°N,45–105°E","Design","Operational preprocessing required"],["Embedding engine","CNN + attention + temporal model","Prototype","Train on GLORYS"],["Depth decoder","15 target levels","Prototype","Train/evaluate"],["Physics loss","surface + vertical constraints","Prototype","Tune λ weights"],["Uncertainty","MC dropout / ensemble","Prototype","Calibrate on validation"],["ARGO validation","Independent INCOIS/Gridded ARGO","Required","External validation dataset"],["Operational API","Daily product + visualization","Planned","Deployment stage"]],columns=["Module","Scope","Status","Next step"])
    st.dataframe(roadmap,use_container_width=True,hide_index=True)
    with st.expander("🔎 Scientific boundary of the prototype"):
        st.write("The included emulator demonstrates the intended user experience and algorithmic workflow. Its numerical outputs are not GLORYS-trained predictions and the ARGO panel is synthetic. The final SIH system must train on GLORYS and validate independently against INCOIS/Gridded ARGO before scientific claims are made.")

st.markdown('<div style="text-align:center;color:#536c7e;font-size:.72rem;margin-top:28px">OCEANEMBED · NORTH INDIAN OCEAN · SIH26066 · PROTOTYPE MISSION CONTROL</div>',unsafe_allow_html=True)
