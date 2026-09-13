# OceanEmbed-NIO Prototype

Interactive prototype for SIH26066: **OceanEmbed - Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**.

## What is included

- North Indian Ocean interactive controls
- SST, SSS, SSH/SLA, surface currents and winds
- Daily temporal-window concept
- 15 requested depths:
  `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`
- Depth-conditioned reconstruction
- Thermocline estimate
- Ocean Heat Content proxy
- Uncertainty estimation
- Physics diagnostics
- Attention/feature-importance view
- Spatial maps
- ARGO-style validation panel
- CSV and experiment-config export
- Research architecture and production-upgrade notes

## Important scientific note

This public prototype is deliberately lightweight and runs without multi-GB GLORYS/ARGO data. Its reconstruction engine is an **ocean-state emulator** used to demonstrate the complete user experience and software workflow.

It is **not a scientifically validated subsurface temperature product**.

For the real research model, replace the emulator with:

1. Daily 0.25° satellite inputs:
   - SST
   - SSS
   - SSH/SLA
   - U/V surface currents
   - U/V surface winds
2. GLORYS temperature target for training.
3. Independent INCOIS/Gridded ARGO for validation.
4. PyTorch CNN + CBAM + temporal Transformer + learned depth embeddings.
5. Physics-aware training loss.
6. MC Dropout/deep ensemble uncertainty.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy and share

The easiest public deployment is Streamlit Community Cloud:

1. Create a GitHub repository.
2. Upload `app.py`, `oceanembed.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io/
4. Sign in with GitHub.
5. Create app → select the repository → select `app.py` → Deploy.
6. Streamlit will give the app a public `*.streamlit.app` URL.

## Production roadmap

### Phase 1
Replace the emulator with real data ingestion and preprocessing.

### Phase 2
Train the OceanEmbed model on GLORYS.

### Phase 3
Independent ARGO validation.

### Phase 4
Add uncertainty, extreme-event weighting, SHAP, and regional evaluation.

### Phase 5
Operational dashboard and standardized daily output.
