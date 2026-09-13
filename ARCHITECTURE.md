# OceanEmbed-NIO target architecture

Surface satellite observations
    |
    v
QC / harmonization / daily 0.25 degree grid
    |
    v
Missing-data mask + normalization
    |
    v
Spatial CNN encoder
    |
    +--> CBAM channel attention
    |
    +--> CBAM spatial attention
    |
    v
Temporal Transformer
    |
    v
Ocean Embedding Z
    |
    +--> geographic / temporal embeddings
    |
    +--> depth embedding
    |
    v
Depth-aware decoder
    |
    +--> Temperature at 15 standard depths
    +--> Thermocline depth
    +--> uncertainty
    |
    v
Physics-aware objective
    L = L_temperature
      + lambda1 L_surface
      + lambda2 L_vertical
      + lambda3 L_smooth
      + lambda4 L_thermocline
      + lambda5 L_extreme
    |
    v
GLORYS training
    |
    v
Independent ARGO validation
