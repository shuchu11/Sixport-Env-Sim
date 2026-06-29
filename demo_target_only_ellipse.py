# -*- coding: utf-8 -*-
"""
Target-only ellipse-calibration demo.

Scene:
  - one chest target with respiration + heartbeat
  - no static clutter
  - no moving interferer
  - intentional six-port hardware imbalance

This isolates the I/Q ellipse caused by hardware mismatch from multipath.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sixport_radar_sim import (
    SixPortRadarSimulator, SixPortHardware,
    scene_vital_signs_with_multipath,
    vital_signs_motion,
    extract_iq, calibrate_iq, recover_displacement, spectrum,
)


FS = 200.0
DUR = 30.0
F0 = 24e9


def rmse_mm(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) * 1e3


# Non-ideal six-port hardware only. No clutter/interferer is added below.
hw = SixPortHardware(
    phase_err_deg=(0.0, 8.0, -5.0, 6.0),
    responsivity=(1.0, 0.85, 1.10, 0.95),
    nonlinearity=(0.02, 0.0, 0.03, 0.01),
    dc_offset=(0.0, 0.0, 0.0, 0.0),
    noise_v_rms=0.002,
    seed=1,
)

sim = scene_vital_signs_with_multipath(
    f0_hz=F0,
    fs_hz=FS,
    target_distance_m=0.5,
    static_clutter=False,
    moving_interferer=False,
    hw=hw,
)

data = sim.collect(DUR)
t = data["t"]
lam = sim.wavelength_m

x_true = vital_signs_motion(4e-3, 0.30, 0.4e-3, 1.20)(t)
x_true = x_true - np.mean(x_true)

I_raw, Q_raw = extract_iq(data["V"])
I_cal, Q_cal, fit = calibrate_iq(I_raw, Q_raw)

x_raw = recover_displacement(I_raw, Q_raw, lam)
x_cal = recover_displacement(I_cal, Q_cal, lam)

print("=== Target only + hardware imbalance ===")
print(f"raw IQ ellipse center          : ({fit['center'][0]:.4f}, {fit['center'][1]:.4f})")
print(f"raw displacement RMSE          : {rmse_mm(x_raw, x_true):.4f} mm")
print(f"calibrated displacement RMSE   : {rmse_mm(x_cal, x_true):.4f} mm")
print(f"raw correlation                : {np.corrcoef(x_true, x_raw)[0, 1]:.6f}")
print(f"calibrated correlation         : {np.corrcoef(x_true, x_cal)[0, 1]:.6f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

for i in range(4):
    ax[0, 0].plot(t[:400], data["V"][i][:400], lw=0.9, label=f"v{i+1}")
ax[0, 0].set_title("Raw four-detector voltages: target only, non-ideal hardware")
ax[0, 0].set_xlabel("time (s)")
ax[0, 0].set_ylabel("V")
ax[0, 0].legend(ncol=4, fontsize=8)

ax[0, 1].plot(I_raw, Q_raw, ".", ms=2, alpha=0.35, label="raw I/Q")
ax[0, 1].plot(I_cal, Q_cal, ".", ms=2, alpha=0.35, label="calibrated I/Q")
ax[0, 1].axhline(0, color="k", lw=0.4)
ax[0, 1].axvline(0, color="k", lw=0.4)
ax[0, 1].set_title("I/Q plane: hardware imbalance ellipse correction")
ax[0, 1].set_xlabel("I")
ax[0, 1].set_ylabel("Q")
ax[0, 1].set_aspect("equal", "datalim")
ax[0, 1].legend(fontsize=8)

ax[1, 0].plot(t, x_true * 1e3, "k", lw=2, label="ground truth")
ax[1, 0].plot(t, x_raw * 1e3, lw=0.8, alpha=0.8, label="raw estimate")
ax[1, 0].plot(t, x_cal * 1e3, lw=0.8, alpha=0.8, label="calibrated estimate")
ax[1, 0].set_xlim(0, 12)
ax[1, 0].set_title("Recovered chest displacement")
ax[1, 0].set_xlabel("time (s)")
ax[1, 0].set_ylabel("displacement (mm)")
ax[1, 0].legend(fontsize=8)

f_t, m_t = spectrum(x_true, FS)
f_c, m_c = spectrum(x_cal, FS)
ax[1, 1].plot(f_t, m_t * 1e3, "k", lw=2, label="ground truth")
ax[1, 1].plot(f_c, m_c * 1e3, lw=1, label="calibrated estimate")
ax[1, 1].axvline(0.30, color="g", ls="--", lw=0.8, label="resp 0.30 Hz")
ax[1, 1].axvline(1.20, color="b", ls="--", lw=0.8, label="heart 1.20 Hz")
ax[1, 1].set_xlim(0, 2.0)
ax[1, 1].set_title("Displacement spectrum")
ax[1, 1].set_xlabel("Hz")
ax[1, 1].set_ylabel("amplitude (mm)")
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("target_only_ellipse_demo.png", dpi=130)
SixPortRadarSimulator.save(
    data,
    "target_only_ellipse_dataset.npz",
    "target_only_ellipse_dataset.csv",
)

print("saved figure -> target_only_ellipse_demo.png")
print("saved dataset -> target_only_ellipse_dataset.csv / target_only_ellipse_dataset.npz")
