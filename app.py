import json
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oceanembed import (
    DEPTHS, REGIONS, generate_surface_state, reconstruct_profile,
    estimate_thermocline, compute_ohc, uncertainty_estimate,
    physics_diagnostics, synthetic_argo_validation, feature_importance,
)

st.set_page_config(page_title="OceanEmbed | NIO", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# --------------------
# Scientific visual theme
# --------------------
st.markdown("""
<style>

:root{--accent:#238edb;--text:#172532;--muted:#536b7b;--line:#9bb5c7;--panel:#dce8ef;--panel2:#cbdce7;--green:#238b70;--amber:#b47a00;--red:#c94f6b;--lav:#596fc4}
.stApp{background:#8faebe;color:var(--text)}
.block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:3rem}
[data-testid="stHeader"]{background:#8faebe}
[data-testid="stSidebar"]{background:#8eafc3;border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1rem}
h1,h2,h3,h4{color:#172532!important;letter-spacing:-.01em}
p,li{color:#304958}
label,[data-testid="stWidgetLabel"] p{color:#243d4c!important;font-weight:600!important}
.small-note{font-size:.78rem;color:var(--muted)}
.brandline{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding:2px 0 12px;margin-bottom:18px}
.brand{font-size:1.05rem;font-weight:800;letter-spacing:.08em;color:#172532}.brand span{color:var(--accent)}
.status{font-size:.72rem;font-weight:700;color:#176a55;background:#c8e6dc;border:1px solid #79b9a8;border-radius:20px;padding:5px 10px}
.hero{background:linear-gradient(135deg,#dce8ef 0%,#cbdce7 65%,#b8cedb 100%);border:1px solid var(--line);border-left:4px solid var(--accent);padding:20px 24px;border-radius:7px;margin-bottom:14px}
.hero-kicker{font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;color:#238edb;font-weight:800}
.hero h1{font-size:2rem;margin:5px 0 4px}.hero p{margin:0;color:#405867;max-width:1000px}
.meta-strip{display:grid;grid-template-columns:1.3fr 1fr 1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:18px}
.meta-cell{background:#dce8ef;padding:10px 13px}.meta-label{font-size:.64rem;text-transform:uppercase;letter-spacing:.08em;color:#587184}.meta-value{font-weight:750;color:#172532;margin-top:2px}
.section-title{margin:20px 0 8px;font-size:1rem;font-weight:800;color:#203746;border-bottom:1px solid var(--line);padding-bottom:6px}
.info-box{background:#dce8ef;border:1px solid var(--line);border-radius:6px;padding:11px 13px;color:#405867;font-size:.8rem}
.warning{background:#f1e4c5;border:1px solid #c9aa68;border-left:4px solid #b47a00;border-radius:5px;padding:10px 12px;color:#654900;font-size:.8rem}
.good{background:#d1e9e1;border:1px solid #79b9a8;border-left:4px solid #238b70;border-radius:5px;padding:10px 12px;color:#185b49;font-size:.8rem}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:5px}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--line);background:transparent}
.stTabs [data-baseweb="tab"]{color:#405867;font-weight:700;padding:9px 15px;border-radius:0}
.stTabs [aria-selected="true"]{color:#238edb!important;border-bottom:2px solid var(--accent);background:transparent}
[data-testid="stMetric"]{background:#dce8ef;border:1px solid #9bb5c7;padding:10px 12px;border-radius:7px}
[data-testid="stMetricLabel"]{color:#536b7b!important}
[data-testid="stMetricValue"]{color:#172532!important}
hr{border-color:var(--line)!important}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


DEFAULTS=dict(sst=28.0,sss=35.0,ssh=0.05,u_cur=0.10,v_cur=0.05,u_wind=3.0,v_wind=1.0)
for k,v in DEFAULTS.items(): st.session_state.setdefault(k,v)

PRESETS={
    "Custom":None,
    "Calm ocean":dict(sst=28.0,sss=35.0,ssh=0.02,u_cur=0.08,v_cur=0.03,u_wind=2.0,v_wind=.8),
    "Marine heatwave":dict(sst=32.0,sss=34.7,ssh=.25,u_cur=.35,v_cur=.20,u_wind=2.5,v_wind=1.0),
    "Strong mixing":dict(sst=29.0,sss=35.2,ssh=-.12,u_cur=.45,v_cur=-.35,u_wind=12.0,v_wind=7.0),
    "Cyclone-like":dict(sst=30.0,sss=35.1,ssh=-.30,u_cur=.60,v_cur=-.55,u_wind=-15.0,v_wind=12.0),
}

def plot_base(fig,height=430):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#102131",
        plot_bgcolor="#0b1927",
        height=height,
        margin=dict(l=50,r=25,t=35,b=45),
        font=dict(color="#dcebf3"),
        legend=dict(bgcolor="rgba(16,33,49,.88)",bordercolor="#28455a",borderwidth=1)
    )
    fig.update_xaxes(gridcolor="#294457",zerolinecolor="#5a7181",linecolor="#5a7181")
    fig.update_yaxes(gridcolor="#294457",zerolinecolor="#5a7181",linecolor="#5a7181")
    return fig

def metric_row(items):
    cols=st.columns(len(items),gap="small")
    for col,(label,value,helptext) in zip(cols,items):
        with col.container(border=True):
            st.metric(label,value,help=helptext)

def status_box(text,kind="good"):
    st.markdown(f'<div class="{kind}">{text}</div>',unsafe_allow_html=True)

def region_label(name):
    return name.replace("_"," ").title()

# --------------------
# Sidebar controls
# --------------------
st.sidebar.markdown("OceanEmbed")
st.sidebar.caption("North Indian Ocean")
st.sidebar.divider()
region_name=st.sidebar.selectbox("Region",list(REGIONS.keys()),index=0,format_func=region_label)
analysis_date=st.sidebar.date_input("Analysis date",value=date(2026,9,12))
preset=st.sidebar.selectbox("Scenario",list(PRESETS.keys()))
if preset!="Custom" and st.sidebar.button("Apply scenario",use_container_width=True):
    for k,v in PRESETS[preset].items(): st.session_state[k]=v
    st.rerun()
with st.sidebar.expander("Surface observations",expanded=True):
    sst=st.slider("SST (°C)",18.0,34.0,.1,key="sst")
    sss=st.slider("SSS (PSU)",30.0,38.0,.1,key="sss")
    ssh=st.slider("SSH / SLA (m)",-0.8,.8,.01,key="ssh")
    u_cur=st.slider("U current (m/s)",-1.5,1.5,.01,key="u_cur")
    v_cur=st.slider("V current (m/s)",-1.5,1.5,.01,key="v_cur")
    u_wind=st.slider("U wind (m/s)",-20.,20.,.1,key="u_wind")
    v_wind=st.slider("V wind (m/s)",-20.,20.,.1,key="v_wind")
with st.sidebar.expander("Model settings"):
    window=st.select_slider("Temporal window (days)",[3,5,7,14],7)
    attention=st.slider("Attention strength",.2,1.,.75,.05)
    extreme=st.slider("Extreme-event weight",1.,5.,2.,.1)
    runs=st.slider("Uncertainty ensemble runs",10,60,30,5)
    show_unc=st.checkbox("Show uncertainty",True)
with st.sidebar.expander("Display"):
    show_markers=st.checkbox("Map region markers",True)
if st.sidebar.button("Reset surface inputs",use_container_width=True):
    for k,v in DEFAULTS.items(): st.session_state[k]=v
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("")

# --------------------
# Reconstruction
# --------------------
surface=generate_surface_state(region_name,pd.Timestamp(analysis_date),sst,sss,ssh,u_cur,v_cur,u_wind,v_wind,window)
profile=reconstruct_profile(surface,attention,extreme)
thermocline=estimate_thermocline(DEPTHS,profile)
ohcc=compute_ohc(DEPTHS,profile)
unc=uncertainty_estimate(surface,attention,extreme,runs)
physics=physics_diagnostics(DEPTHS,profile,sst)
importance=feature_importance(surface)
mean_unc=float(np.mean(unc))
confidence=max(50,min(99,100-mean_unc*35))
gradient=np.gradient(profile,DEPTHS)
current_speed=float(np.hypot(u_cur,v_cur));wind_speed=float(np.hypot(u_wind,v_wind))
mixing_index=float(np.clip(45+wind_speed*2.7+current_speed*22+abs(ssh)*35,0,100))
stratification=float(np.clip(100-np.mean(np.abs(gradient))*900,0,100))
anomaly_index=float(np.clip((sst-28)*14+ssh*35,-100,100))
if sst>=31 or physics["score"]<60: risk="High"; risk_color="red"
elif sst>=29.5 or mean_unc>.45 or physics["score"]<75: risk="Watch"; risk_color="amber"
else: risk="Stable"; risk_color="green"

# --------------------
# Header
# --------------------
st.markdown('<div class="brandline"><div class="brand">OCEAN<span>EMBED</span> / NIO</div><div class="status"></div></div>',unsafe_allow_html=True)
st.markdown(f'<div class="hero"><div class="hero-kicker">Subsurface temperature reconstruction</div><h1>North Indian Ocean analysis</h1><p>Daily 0.25° concept for reconstructing depth-wise ocean temperature from satellite surface observations. The interface is organized around observations, reconstruction, spatial structure, validation and model diagnostics.</p></div>',unsafe_allow_html=True)
st.markdown(f'''<div class="meta-strip"><div class="meta-cell"><div class="meta-label">Region</div><div class="meta-value">{region_label(region_name)}</div></div><div class="meta-cell"><div class="meta-label">Date</div><div class="meta-value">{analysis_date}</div></div><div class="meta-cell"><div class="meta-label">Scenario</div><div class="meta-value">{preset}</div></div><div class="meta-cell"><div class="meta-label">State</div><div class="meta-value">{risk}</div></div></div>''',unsafe_allow_html=True)

st.markdown('<div class="section-title">Current reconstruction</div>',unsafe_allow_html=True)
metric_row([
    ("Thermocline",f"{thermocline:.1f} m","Estimated from the reconstructed vertical profile"),
    ("OHC proxy",f"{ohcc:,.0f} MJ/m²","Integrated heat-content proxy from the profile"),
    ("Mean uncertainty",f"±{mean_unc:.2f} °C","Mean emulator uncertainty across target depths"),
    ("Physics score",f"{physics['score']:.1f}/100","Diagnostic score"),
    ("Confidence",f"{confidence:.0f}%","Derived confidence indicator"),
])

# --------------------
# Main tabs
# --------------------
T=st.tabs(["Overview","Profile","Spatial field","Validation","Diagnostics","Scenarios","Data & methods"])

# Overview
with T[0]:
    left,right=st.columns([1.55,.9],gap="large")
    with left:
        st.markdown('<div class="section-title">Reconstructed vertical temperature</div>',unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=profile,y=DEPTHS,mode="lines+markers",name="OceanEmbed",line=dict(color="#66d9ff",width=3),marker=dict(size=6)))
        if show_unc:
            fig.add_trace(go.Scatter(x=profile+unc,y=DEPTHS,line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=profile-unc,y=DEPTHS,fill="tonexty",fillcolor="rgba(102,217,255,.16)",line=dict(width=0),name="Uncertainty"))
        fig.add_hline(y=thermocline,line_dash="dot",line_color="#ffd166",annotation_text=f"Thermocline {thermocline:.0f} m")
        fig.update_yaxes(autorange="reversed",title="Depth (m)")
        fig.update_xaxes(title="Temperature (°C)")
        plot_base(fig,500);st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Surface observations</div>',unsafe_allow_html=True)
        obs=pd.DataFrame({"Variable":["SST","SSS","SSH / SLA","U current","V current","U wind","V wind"],"Value":[f"{sst:.1f} °C",f"{sss:.1f} PSU",f"{ssh:.2f} m",f"{u_cur:.2f} m/s",f"{v_cur:.2f} m/s",f"{u_wind:.1f} m/s",f"{v_wind:.1f} m/s"]})
        st.dataframe(obs,use_container_width=True,hide_index=True,height=295)
        status_box(f"Current assessment: <b>{risk}</b>","warning" if risk!="Stable" else "good")
        st.markdown('<div class="section-title">Derived indicators</div>',unsafe_allow_html=True)
        metric_row([("Mixing index",f"{mixing_index:.0f}/100","Proxy from winds, currents and SSH"),("Stratification",f"{stratification:.0f}/100","Profile-gradient proxy"),("Surface anomaly",f"{anomaly_index:+.1f}","Relative to baseline")])
    st.markdown('<div class="section-title"></div>',unsafe_allow_html=True)
    

# Profile
with T[1]:
    d1,d2,d3=st.columns([1,.7,.7],gap="medium")
    selected_depth=d1.select_slider("Inspection depth (m)",list(DEPTHS),value=100)
    temp=float(np.interp(selected_depth,DEPTHS,profile));sigma=float(np.interp(selected_depth,DEPTHS,unc))
    d2.metric("Temperature",f"{temp:.2f} °C")
    d3.metric("Uncertainty",f"±{sigma:.2f} °C")
    p1,p2=st.columns([1.5,.85],gap="large")
    with p1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=profile,y=DEPTHS,mode="lines+markers",name="Reconstructed",line=dict(color="#66d9ff",width=3)))
        if show_unc:
            fig.add_trace(go.Scatter(x=profile+unc,y=DEPTHS,line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=profile-unc,y=DEPTHS,fill="tonexty",fillcolor="rgba(102,217,255,.16)",line=dict(width=0),name="± uncertainty"))
        fig.add_trace(go.Scatter(x=[temp],y=[selected_depth],mode="markers",marker=dict(size=11,color="#ffd166"),name="Selected depth"))
        fig.add_hline(y=thermocline,line_dash="dot",line_color="#ffd166")
        fig.update_yaxes(autorange="reversed",title="Depth (m)");fig.update_xaxes(title="Temperature (°C)")
        plot_base(fig,600);st.plotly_chart(fig,use_container_width=True)
    with p2:
        gf=go.Figure(go.Bar(x=gradient,y=DEPTHS,orientation="h",marker_color="#7ee2c3"))
        gf.add_vline(x=0,line_dash="dot",line_color="#9aaab4");gf.update_yaxes(autorange="reversed",title="Depth (m)");gf.update_xaxes(title="dT/dz (°C/m)")
        plot_base(gf,410);st.plotly_chart(gf,use_container_width=True)
        st.dataframe(pd.DataFrame({"Depth (m)":DEPTHS,"Temperature (°C)":np.round(profile,3),"Uncertainty (°C)":np.round(unc,3),"dT/dz":np.round(gradient,5)}),use_container_width=True,hide_index=True,height=250)

# Spatial
with T[2]:
    c1,c2,c3=st.columns([.8,1,1],gap="medium")
    map_depth=c1.select_slider("Depth (m)",list(DEPTHS),value=100,key="spatial_depth")
    field_mode=c2.selectbox("Variable",["Temperature","Uncertainty","Anomaly","Gradient proxy"])
    show_map_markers=c3.checkbox("Show regional reference points",show_markers)
    lon=np.linspace(45,105,121);lat=np.linspace(5,30,101);LON,LAT=np.meshgrid(lon,lat)
    dt=float(np.interp(map_depth,DEPTHS,profile))
    spatial=(sst+1.4*np.sin((LON-65)/8)-1.1*np.cos((LAT-17)/5)+ssh*2+.8*np.sin((LON+LAT)/9))-(sst-dt)*(0.55+.35*np.exp(-map_depth/250))
    uf=mean_unc*(.75+.4*np.abs(np.sin((LON-60)/12))*np.cos((LAT-17)/7)**2)
    af=spatial-np.mean(spatial);gf=np.gradient(spatial,axis=0)
    field=spatial if field_mode=="Temperature" else uf if field_mode=="Uncertainty" else af if field_mode=="Anomaly" else gf
    colors="RdBu_r" if field_mode in ["Anomaly","Gradient proxy"] else "Turbo" if field_mode=="Temperature" else "Viridis"
    mf=go.Figure(go.Heatmap(x=lon,y=lat,z=field,colorscale=colors,colorbar=dict(title=field_mode),hovertemplate="Lon %{x:.2f}°E<br>Lat %{y:.2f}°N<br>%{z:.2f}<extra></extra>"))
    if show_map_markers:
        mf.add_trace(go.Scatter(x=[REGIONS[r]["lon"] for r in REGIONS],y=[REGIONS[r]["lat"] for r in REGIONS],text=[region_label(r) for r in REGIONS],mode="markers+text",textposition="top center",marker=dict(size=7,color="#d6e8f0",line=dict(color="#08131d",width=1)),name="Regions"))
    mf.update_xaxes(title="Longitude (°E)");mf.update_yaxes(title="Latitude (°N)");plot_base(mf,650);st.plotly_chart(mf,use_container_width=True)
    metric_row([("Depth",f"{map_depth} m","Selected vertical layer"),("Spatial mean",f"{np.mean(field):.2f}","Mean of displayed field"),("Spatial spread",f"{np.std(field):.2f}","Standard deviation of displayed field")])
    status_box("","warning")

# Validation
with T[3]:
    argo=synthetic_argo_validation(profile,uncertainty=unc,seed=42)
    st.markdown('<div class="warning"></div>',unsafe_allow_html=True)
    metric_row([("RMSE",f"{argo['rmse']:.2f} °C","Synthetic reference"),("MAE",f"{argo['mae']:.2f} °C","Synthetic reference"),("R²",f"{argo['r2']:.4f}","Synthetic reference"),("Bias",f"{argo['bias']:+.2f} °C","Mean signed error")])
    v1,v2=st.columns([1.2,1],gap="large")
    with v1:
        vf=go.Figure(go.Scatter(x=argo["observed"],y=argo["predicted"],mode="markers",marker=dict(size=9,color="#66d9ff"),customdata=DEPTHS,hovertemplate="%{customdata:.0f} m<br>Reference %{x:.2f} °C<br>Prediction %{y:.2f} °C<extra></extra>"))
        mn=min(argo["observed"].min(),argo["predicted"].min());mx=max(argo["observed"].max(),argo["predicted"].max())
        vf.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",line=dict(color="#ffd166",dash="dash"),name="1:1"));vf.update_xaxes(title="Reference temperature (°C)");vf.update_yaxes(title="Predicted temperature (°C)");plot_base(vf,470);st.plotly_chart(vf,use_container_width=True)
    with v2:
        residual=argo["predicted"]-argo["observed"]
        rf=go.Figure(go.Bar(x=DEPTHS,y=residual,marker_color="#7ee2c3"));rf.add_hline(y=0,line_dash="dot");rf.update_xaxes(title="Depth (m)");rf.update_yaxes(title="Residual (°C)");plot_base(rf,470);st.plotly_chart(rf,use_container_width=True)

# Diagnostics
with T[4]:
    a,b=st.columns([1.25,.9],gap="large")
    with a:
        imp=importance.sort_values("Importance")
        fig=go.Figure(go.Bar(x=imp.Importance,y=imp.Feature,orientation="h",marker_color="#66d9ff"));fig.update_xaxes(title="Relative contribution");plot_base(fig,430);st.plotly_chart(fig,use_container_width=True)
    with b:
        latent=np.array([np.mean(profile),np.std(profile),thermocline,ohcc/100,mean_unc,physics["score"],mixing_index,stratification])
        labels=["Mean T","T spread","Thermocline","OHC","Uncertainty","Physics","Mixing","Stratification"]
        lo,hi=latent.min(),latent.max();norm=np.zeros_like(latent) if hi==lo else (latent-lo)/(hi-lo)
        radar=go.Figure(go.Scatterpolar(r=norm,theta=labels,fill="toself",line=dict(color="#66d9ff",width=2),fillcolor="rgba(102,217,255,.16)"));radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor="#2a3e4d"),bgcolor="#0b1620"));plot_base(radar,430);st.plotly_chart(radar,use_container_width=True)
    st.markdown('<div class="section-title">Vertical diagnostics</div>',unsafe_allow_html=True)
    af=go.Figure(go.Heatmap(z=np.array([np.abs(gradient)*.65+unc/np.max(unc)*.35]),x=DEPTHS,y=["diagnostic weight"],colorscale="Blues",colorbar=dict(title="relative"),hovertemplate="Depth %{x:.0f} m<br>Weight %{z:.2f}<extra></extra>"));af.update_xaxes(title="Depth (m)");plot_base(af,240);st.plotly_chart(af,use_container_width=True)
    metric_row([("Physics score",f"{physics['score']:.1f}/100","Physics diagnostic"),("Mixing index",f"{mixing_index:.0f}/100","Wind/current proxy"),("Stratification",f"{stratification:.0f}/100","Vertical-gradient proxy"),("Attention state",risk,"operational state")])

# Scenarios
with T[5]:
    selected=st.multiselect("Compare scenarios",list(PRESETS.keys())[1:],default=list(PRESETS.keys())[1:4],max_selections=4)
    fig=go.Figure();records=[]
    for name in selected:
        v=PRESETS[name];ss=generate_surface_state(region_name,pd.Timestamp(analysis_date),v["sst"],v["sss"],v["ssh"],v["u_cur"],v["v_cur"],v["u_wind"],v["v_wind"],window);pp=reconstruct_profile(ss,attention,extreme)
        fig.add_trace(go.Scatter(x=pp,y=DEPTHS,mode="lines+markers",name=name));records.append([name,v["sst"],estimate_thermocline(DEPTHS,pp),compute_ohc(DEPTHS,pp)])
    fig.update_yaxes(autorange="reversed",title="Depth (m)");fig.update_xaxes(title="Temperature (°C)");plot_base(fig,520);st.plotly_chart(fig,use_container_width=True)
    if records: st.dataframe(pd.DataFrame(records,columns=["Scenario","SST (°C)","Thermocline (m)","OHC (MJ/m²)"]).round(2),use_container_width=True,hide_index=True)
    st.markdown('<div class="section-title">Sensitivity analysis</div>',unsafe_allow_html=True)
    s1,s2=st.columns([1,.6]);
    with s1: sweep_feature=st.selectbox("Input variable",["SST","SSS","SSH/SLA","Wind speed"])
    with s2: sweep_points=st.slider("Number of points",7,25,13)
    if sweep_feature=="SST": sweep=np.linspace(max(18,sst-3),min(34,sst+3),sweep_points)
    elif sweep_feature=="SSS": sweep=np.linspace(max(30,sss-1.5),min(38,sss+1.5),sweep_points)
    elif sweep_feature=="SSH/SLA": sweep=np.linspace(max(-.8,ssh-.35),min(.8,ssh+.35),sweep_points)
    else: sweep=np.linspace(max(0,wind_speed-8),min(28,wind_speed+8),sweep_points)
    outs=[]
    for val in sweep:
        vals=dict(sst=sst,sss=sss,ssh=ssh,u_cur=u_cur,v_cur=v_cur,u_wind=u_wind,v_wind=v_wind)
        if sweep_feature=="SST": vals["sst"]=float(val)
        elif sweep_feature=="SSS": vals["sss"]=float(val)
        elif sweep_feature=="SSH/SLA": vals["ssh"]=float(val)
        else: vals["u_wind"]=float(val);vals["v_wind"]=0
        ss=generate_surface_state(region_name,pd.Timestamp(analysis_date),vals["sst"],vals["sss"],vals["ssh"],vals["u_cur"],vals["v_cur"],vals["u_wind"],vals["v_wind"],window);pp=reconstruct_profile(ss,attention,extreme)
        outs.append([val,estimate_thermocline(DEPTHS,pp),compute_ohc(DEPTHS,pp)])
    sw=pd.DataFrame(outs,columns=[sweep_feature,"Thermocline","OHC"])
    sf=go.Figure();sf.add_trace(go.Scatter(x=sw[sweep_feature],y=sw.Thermocline,mode="lines+markers",name="Thermocline"));sf.add_trace(go.Scatter(x=sw[sweep_feature],y=sw.OHC,mode="lines+markers",name="OHC",yaxis="y2"));sf.update_layout(yaxis=dict(title="Thermocline (m)"),yaxis2=dict(title="OHC (MJ/m²)",overlaying="y",side="right"));plot_base(sf,430);st.plotly_chart(sf,use_container_width=True)

# Data and methods
with T[6]:
    export_df=pd.DataFrame({"date":[str(analysis_date)]*len(DEPTHS),"region":[region_name]*len(DEPTHS),"depth_m":DEPTHS,"temperature_c":profile,"uncertainty_c":unc,"dT_dz_c_per_m":gradient})
    config={"date":str(analysis_date),"region":region_name,"scenario":preset,"window_days":window,"attention_strength":attention,"extreme_weight":extreme,"uncertainty_runs":runs,"surface_inputs":{"SST_C":sst,"SSS_PSU":sss,"SSH_m":ssh,"U_current_mps":u_cur,"V_current_mps":v_cur,"U_wind_mps":u_wind,"V_wind_mps":v_wind},"outputs":{"thermocline_m":thermocline,"ohc_proxy_MJ_m2":ohcc,"mean_uncertainty_C":mean_unc,"physics_score":physics["score"],"confidence_percent":confidence,"risk_state":risk}}
    st.markdown('<div class="section-title">Export current experiment</div>',unsafe_allow_html=True)
    x1,x2,x3=st.columns(3)
    with x1: st.download_button("Download temperature CSV",export_df.to_csv(index=False),"oceanembed_profile.csv","text/csv",use_container_width=True)
    with x2: st.download_button("Download experiment JSON",json.dumps(config,indent=2),"oceanembed_experiment.json","application/json",use_container_width=True)
    with x3: st.download_button("Download report TXT",("OCEANEMBED NIO REPORT\n\n"+json.dumps(config,indent=2)),"oceanembed_report.txt","text/plain",use_container_width=True)
    st.dataframe(export_df.round(4),use_container_width=True,hide_index=True)
    st.markdown('<div class="section-title">Implementation status</div>',unsafe_allow_html=True)
    roadmap=pd.DataFrame([
        ["Satellite inputs","SST / SSS / SSH / U / V / winds","Prototype","Operational ingest + QC"],
        ["Daily 0.25° NIO grid","5–30°N, 45–105°E","Design","Harmonization pipeline"],
        ["Embedding engine","CNN + attention + temporal model","Prototype","Train on GLORYS"],
        ["Depth decoder","15 target levels","Prototype","Train + evaluate"],
        ["Physics-aware loss","Surface + vertical constraints","Prototype","Tune loss weights"],
        ["Uncertainty","MC dropout / ensemble","Prototype","Calibrate"],
        ["Independent validation","INCOIS / Gridded ARGO","Required","Integrate validation set"],
        ["Operational product","Daily reconstruction + map","Planned","Deployment stage"],
    ],columns=["Component","Scope","Current status","Next step"])
    st.dataframe(roadmap,use_container_width=True,hide_index=True)
    with st.expander("Scientific scope and limitations"):
        st.write(".")

st.markdown('<div style="text-align:center;color:#7b8a93;font-size:.72rem;margin-top:30px">OceanEmbed · North Indian Ocean ·</div>',unsafe_allow_html=True)
