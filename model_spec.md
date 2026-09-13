# Production model specification

## Input tensor

Recommended training window:

`X ∈ R^(B, T, C, H, W)`

where:
- B = batch
- T = temporal window, e.g. 7 days
- C = 7 surface variables
- H/W = daily 0.25° North Indian Ocean grid

Channels:
1. SST
2. SSS
3. SSH/SLA
4. U current
5. V current
6. U wind
7. V wind

Add a missing-data mask as additional channels.

## Encoder

CNN blocks -> CBAM -> spatial feature map.

## Temporal module

Flatten/project spatial tokens and process the T-day sequence with a Transformer encoder.

## Ocean embedding

Project the fused representation to a compact latent vector Z.

## Depth decoder

For each requested depth d:

`h_d = MLP([Z, DepthEmbedding(d), GeoEmbedding, TimeEmbedding])`

`T_d = TemperatureHead(h_d)`

## Auxiliary head

`ThermoclineHead(Z) -> thermocline depth`

## Uncertainty

Use MC Dropout or a small deep ensemble during inference.

## Loss

`L_total = L_temp + λ1 L_surface + λ2 L_vertical + λ3 L_smooth + λ4 L_thermocline + λ5 L_extreme`

## Evaluation

Report:
- RMSE
- MAE
- Bias
- R²
- correlation
- depth-wise errors
- seasonal errors
- regional errors
- thermocline error
- uncertainty calibration

Validate independently with ARGO rather than only against the reanalysis target.
