"""
generate_dataset.py
-------------------
Generates a synthetic, physically-inspired FinFET dataset for the ML demo.

IMPORTANT: This data is SYNTHETIC. It is not measured or TCAD-simulated data.
It uses simplified physics equations to capture qualitative FinFET behavior.
"""

import numpy as np
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
N = 6000          # total samples

# ── 1. Randomly sample device and operating parameters ────────────────────────
Lg    = np.random.uniform(7,   50,  N)   # Gate length          (nm)
Wfin  = np.random.uniform(4,   15,  N)   # Fin width            (nm)
Hfin  = np.random.uniform(20,  60,  N)   # Fin height           (nm)
Tox   = np.random.uniform(0.8,  3,  N)   # Oxide thickness      (nm)
Vth   = np.random.uniform(0.2, 0.5, N)   # Threshold voltage    (V)
VGS   = np.random.uniform(0,   1.2, N)   # Gate-source voltage  (V)
VDS   = np.random.uniform(0,   1.0, N)   # Drain-source voltage (V)
Temp  = np.random.uniform(200, 400, N)   # Temperature          (K)

# ── 2. Physical model constants ───────────────────────────────────────────────
T0    = 300.0          # Reference temperature (K)
mu0   = 400e-4         # Reference mobility  (m²/V·s) – approximate for Si electrons
eps0  = 8.854e-12      # Permittivity of free space
eps_ox = 3.9 * eps0    # SiO2 permittivity

# ── 3. Derived quantities ─────────────────────────────────────────────────────
Tox_m  = Tox  * 1e-9  # nm → m
Lg_m   = Lg   * 1e-9
Wfin_m = Wfin * 1e-9
Hfin_m = Hfin * 1e-9

# Effective width (FinFET – gate wraps three sides of fin)
Weff = Wfin_m + 2 * Hfin_m

# Gate oxide capacitance per unit area  Cox = ε_ox / Tox
Cox = eps_ox / Tox_m   # F/m²

# Temperature-dependent mobility  μ ∝ (T/T0)^(-1.5)  (simplified)
mu = mu0 * (Temp / T0) ** (-1.5)

# Short-channel effect factor: shorter gate → higher current
SCE_factor = 1.0 + 0.3 * np.exp(-Lg / 20.0)

# ── 4. Drain current (Id) model ───────────────────────────────────────────────
Vov = VGS - Vth          # overdrive voltage
Vov_eff = np.maximum(Vov, 0.0)   # clamp to 0 below threshold

# Saturation voltage
Vdsat = Vov_eff          # simplified: Vdsat ≈ Vov

# Linear-region factor: min(VDS, Vdsat)
VDS_eff = np.minimum(VDS, Vdsat)

# Long-channel MOSFET Id (above threshold):
#   Id = μ·Cox·(Weff/Lg)·[(Vov·VDS_eff) - (VDS_eff²/2)]·SCE
Id_above = (mu * Cox * (Weff / Lg_m)
            * (Vov_eff * VDS_eff - 0.5 * VDS_eff ** 2)
            * SCE_factor)

# Sub-threshold current (below threshold):
#   Id_sub = I0·exp(Vov / n·Vt)·(1 − exp(−VDS/Vt))
n  = 1.3           # ideality factor
Vt = 0.02585       # thermal voltage at 300 K (V)
Vt_T = Vt * (Temp / T0)  # temperature-scaled Vt
I0 = 1e-9 * (Weff / Lg_m) * SCE_factor
Id_sub = I0 * np.exp(Vov / (n * Vt_T)) * (1 - np.exp(-VDS / Vt_T))

# Smoothly blend above-threshold and sub-threshold:
#   use sigmoid to avoid hard discontinuity
sigma = 1.0 / (1.0 + np.exp(-(Vov - 0.05) / 0.02))
Id = sigma * Id_above + (1 - sigma) * Id_sub

# Ensure Id ≥ 0
Id = np.maximum(Id, 0.0)

# ── 5. Add realistic noise (0.5 % std) ───────────────────────────────────────
noise = np.random.normal(0, 0.005, N)
Id = Id * (1 + noise)
Id = np.maximum(Id, 0.0)

# ── 6. Assemble DataFrame and save ───────────────────────────────────────────
df = pd.DataFrame({
    "Lg_nm":   Lg,
    "Wfin_nm": Wfin,
    "Hfin_nm": Hfin,
    "Tox_nm":  Tox,
    "Vth_V":   Vth,
    "VGS_V":   VGS,
    "VDS_V":   VDS,
    "Temp_K":  Temp,
    "Id_A":    Id,
})

df.to_csv("dataset.csv", index=False)
print(f"Dataset saved: {len(df)} samples → dataset.csv")
print(f"Id range:  {Id.min():.4e} A  to  {Id.max():.4e} A")
print(f"Id median: {np.median(Id):.4e} A")
