import numpy as np
import pandas as pd

DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)

REGIONS = {
    "Bay of Bengal": {"lat": 15.0, "lon": 88.0, "factor": 1.00},
    "Arabian Sea": {"lat": 15.0, "lon": 65.0, "factor": 0.92},
    "Equatorial Indian Ocean": {"lat": 6.0, "lon": 75.0, "factor": 1.05},
    "North Indian Ocean": {"lat": 18.0, "lon": 75.0, "factor": 1.00},
}

FEATURES = ["SST", "SSS", "SSH/SLA", "U current", "V current", "U wind", "V wind"]

def generate_surface_state(region, date, sst, sss, ssh, u_cur, v_cur, u_wind, v_wind, window=7):
    rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")) % (2**32-1))
    t = np.arange(window)
    seasonal = 0.35*np.sin(2*np.pi*(t+1)/30.0)
    return {
        "region": region,
        "date": pd.Timestamp(date),
        "window": window,
        "SST": float(sst) + seasonal,
        "SSS": float(sss) + 0.08*seasonal,
        "SSH": float(ssh) + 0.03*rng.normal(size=window),
        "U current": float(u_cur) + 0.12*rng.normal(size=window),
        "V current": float(v_cur) + 0.12*rng.normal(size=window),
        "U wind": float(u_wind) + 0.8*rng.normal(size=window),
        "V wind": float(v_wind) + 0.8*rng.normal(size=window),
        "rng": rng,
    }

def _profile_formula(sst, sss, ssh, uc, vc, uw, vw, factor=1.0):
    # Lightweight ocean-state emulator used only so the public prototype works
    # without downloading multi-GB GLORYS/ARGO datasets.
    warm = 0.35*(sst-26.0)
    sal = -0.10*(sss-35.0)
    dynamic = 0.9*ssh + 0.35*np.hypot(uc, vc)
    wind_mix = 0.018*np.hypot(uw, vw)
    surface = sst + 0.35*warm + 0.15*sal
    mixed_layer = 38 + 15*factor + 7*wind_mix - 8*ssh
    thermocline = np.clip(mixed_layer + 42 - 7*ssh - 2*warm, 55, 145)
    deep = 4.0 + 0.018*(sst-25) - 0.08*(sss-35) + 0.20*dynamic
    decay = np.exp(-DEPTHS/220)
    therm = 1/(1+np.exp((DEPTHS-thermocline)/18))
    profile = deep + (surface-deep)*(0.20*decay + 0.80*therm)
    profile -= wind_mix * (1-np.exp(-DEPTHS/80))
    profile += 0.25*np.sin(DEPTHS/140)*np.exp(-DEPTHS/600)
    profile[0] = sst
    return profile, thermocline

def reconstruct_profile(surface, attention_strength=0.75, extreme_weight=2.0):
    sst = float(np.mean(surface["SST"][-3:]))
    sss = float(np.mean(surface["SSS"][-3:]))
    ssh = float(np.mean(surface["SSH"][-3:]))
    uc = float(np.mean(surface["U current"][-3:]))
    vc = float(np.mean(surface["V current"][-3:]))
    uw = float(np.mean(surface["U wind"][-3:]))
    vw = float(np.mean(surface["V wind"][-3:]))

    factor = REGIONS[surface["region"]]["factor"]
    p, thermo = _profile_formula(sst, sss, ssh, uc, vc, uw, vw, factor)

    # Attention proxy: stronger SSH/current influence under dynamically active states.
    activity = np.clip(abs(ssh)*1.4 + np.hypot(uc,vc)*0.25, 0, 1)
    p += attention_strength * activity * 0.35 * np.exp(-DEPTHS/350)
    p -= (extreme_weight-1)*0.04*np.maximum(0, abs(sst-29.0)) * np.exp(-DEPTHS/180)
    p[0] = sst
    return p

def estimate_thermocline(depths, profile):
    grad = np.gradient(profile, depths)
    # Ignore surface interval and choose strongest negative gradient.
    valid = (depths >= 30) & (depths <= 300)
    idx = np.where(valid)[0][np.argmin(grad[valid])]
    return float(depths[idx])

def compute_ohc(depths, profile):
    # Simplified heat-content proxy. Operational OHC requires density, Cp,
    # reference temperature and physically consistent units.
    rho_cp = 4.0e6
    tref = 26.0
    temp_anom = np.maximum(profile - tref, 0)
    integral = np.trapezoid(temp_anom, depths)
    return float(rho_cp * integral / 1e6)

def uncertainty_estimate(surface, attention_strength, extreme_weight, n_runs=30):
    rng = np.random.default_rng(12345)
    base = reconstruct_profile(surface, attention_strength, extreme_weight)
    sst = float(np.mean(surface["SST"][-3:]))
    dynamic = abs(float(np.mean(surface["SSH"][-3:]))) + 0.25*np.hypot(
        float(np.mean(surface["U current"][-3:])),
        float(np.mean(surface["V current"][-3:])),
    )
    base_sigma = 0.18 + 0.18*dynamic + 0.12*np.exp(DEPTHS/850)
    draws = []
    for _ in range(n_runs):
        draws.append(base + rng.normal(0, base_sigma))
    return np.std(np.vstack(draws), axis=0)

def physics_diagnostics(depths, profile, sst):
    surface_error = abs(float(profile[0])-float(sst))
    grad = np.gradient(profile, depths)
    second = np.gradient(grad, depths)
    surface_consistency = max(0, 100 - surface_error*100)
    smoothness = max(0, 100 - np.mean(np.abs(second))*3000)
    realistic = max(0, 100 - np.mean(np.maximum(0, grad))*800)
    score = 0.4*surface_consistency + 0.3*smoothness + 0.3*realistic
    return {
        "surface_consistency": float(surface_consistency),
        "smoothness": float(smoothness),
        "gradient_realism": float(realistic),
        "score": float(score),
    }

def feature_importance(surface):
    vals = np.array([
        np.std(surface["SST"]) + abs(np.mean(surface["SST"])-27)*0.03,
        np.std(surface["SSS"]) + 0.04,
        abs(np.mean(surface["SSH"])) + 0.06,
        np.hypot(np.mean(surface["U current"]), np.mean(surface["V current"])) + 0.05,
        np.hypot(np.mean(surface["U current"]), np.mean(surface["V current"])) + 0.04,
        np.hypot(np.mean(surface["U wind"]), np.mean(surface["V wind"]))*0.025 + 0.03,
        np.hypot(np.mean(surface["U wind"]), np.mean(surface["V wind"]))*0.02 + 0.02,
    ])
    vals = vals / vals.sum()
    return pd.DataFrame({"Feature": FEATURES, "Importance": vals})

def synthetic_argo_validation(profile, uncertainty=None, seed=42):
    rng = np.random.default_rng(seed)
    n = len(profile)
    observed = profile + rng.normal(0, 0.22, n)
    predicted = profile + rng.normal(0, 0.12, n)
    err = predicted-observed
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    ss_res = np.sum((observed-predicted)**2)
    ss_tot = np.sum((observed-np.mean(observed))**2)
    r2 = 1-ss_res/ss_tot if ss_tot else 0.0
    return {
        "observed": observed,
        "predicted": predicted,
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "r2": float(r2),
    }
