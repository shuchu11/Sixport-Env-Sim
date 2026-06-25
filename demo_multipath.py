# -*- coding: utf-8 -*-
"""
demo_multipath.py
=================
Demonstrates the six-port radar simulator on a multipath-interference scene
and runs the baseline mitigation pipeline. Produces:
  - sixport_demo.png       (4-panel figure)
  - sixport_dataset.csv    (raw 4-detector "experimental data")
  - sixport_dataset.npz    (same + ground truth, for offline algorithm dev)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sixport_radar_sim import (
    SixPortRadarSimulator, SixPortHardware, Path,
    static_motion, sinusoid_motion, vital_signs_motion,
    scene_vital_signs_with_multipath,
    extract_iq, calibrate_iq, recover_displacement, spectrum,
)

FS = 200.0          # Hz
DUR = 30.0          # s
F0 = 24e9           # 24 GHz K-band

# ----- Scene A: clean (main target only) -- the reference -------------------
hw = SixPortHardware(phase_err_deg=(0, 8, -5, 6),
                     responsivity=(1.0, 0.85, 1.1, 0.95),
                     nonlinearity=(0.02, 0.0, 0.03, 0.01),
                     noise_v_rms=0.002, seed=1)

sim_clean = SixPortRadarSimulator(f0_hz=F0, fs_hz=FS, hw=hw)
sim_clean.add_path(Path(0.5, 1.0,
                        vital_signs_motion(4e-3, 0.30, 0.4e-3, 1.20),
                        name="chest"))

# ----- Scene B: same target + STATIC multipath + MOVING interferer ----------
sim_mp = scene_vital_signs_with_multipath(
    f0_hz=F0, fs_hz=FS, target_distance_m=0.5,
    static_clutter=True, moving_interferer=True, hw=hw)

data_clean = sim_clean.collect(DUR)
data_mp = sim_mp.collect(DUR)
lam = sim_mp.wavelength_m

# ----- Ground-truth chest displacement (for error evaluation) ---------------
t = data_mp["t"]
x_true = vital_signs_motion(4e-3, 0.30, 0.4e-3, 1.20)(t)
x_true = x_true - np.mean(x_true)

# ----- Process CLEAN scene --------------------------------------------------
I0, Q0 = extract_iq(data_clean["V"])
I0c, Q0c, _ = calibrate_iq(I0, Q0)
x_clean = recover_displacement(I0c, Q0c, lam)

# ----- Process MULTIPATH scene ----------------------------------------------
I1, Q1 = extract_iq(data_mp["V"])
# (a) no mitigation -- just demodulate the corrupted IQ
x_raw = recover_displacement(I1, Q1, lam)
# (b) baseline mitigation: ellipse calibration removes the STATIC-multipath DC
I1c, Q1c, fit = calibrate_iq(I1, Q1)
x_cal = recover_displacement(I1c, Q1c, lam)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) * 1e3  # mm

print("=== Displacement RMSE vs ground truth (mm) ===")
print(f" clean scene, calibrated      : {rmse(x_clean, x_true):.3f}")
print(f" multipath, NO mitigation     : {rmse(x_raw,   x_true):.3f}")
print(f" multipath, ellipse-calibrated: {rmse(x_cal,   x_true):.3f}")
print(f" ellipse center (DC offset)   : ({fit['center'][0]:.3f}, {fit['center'][1]:.3f})")

# ----- Figure ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (1) Raw four-detector voltages -- the measured "experimental data"
for i in range(4):
    ax[0, 0].plot(t[:400], data_mp["V"][i][:400], lw=0.9, label=f"v{i+1}")
ax[0, 0].set_title("Raw four-detector voltages (measured data)")
ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("V"); ax[0, 0].legend(ncol=4, fontsize=8)

# (2) IQ plane: clean vs multipath(raw) vs multipath(calibrated)
ax[0, 1].plot(I0, Q0, '.', ms=2, alpha=0.4, label="clean (raw)")
ax[0, 1].plot(I1, Q1, '.', ms=2, alpha=0.4, label="multipath (raw)")
ax[0, 1].plot(I1c, Q1c, '.', ms=2, alpha=0.4, label="multipath (calibrated)")
ax[0, 1].axhline(0, color='k', lw=0.4); ax[0, 1].axvline(0, color='k', lw=0.4)
ax[0, 1].set_title("I/Q plane: multipath distorts & offsets the arc")
ax[0, 1].set_xlabel("I"); ax[0, 1].set_ylabel("Q")
ax[0, 1].set_aspect("equal", "datalim"); ax[0, 1].legend(fontsize=8)

# (3) Recovered displacement
ax[1, 0].plot(t, x_true * 1e3, 'k', lw=2, label="ground truth")
ax[1, 0].plot(t, x_raw * 1e3, lw=0.8, alpha=0.8, label="multipath, no mitigation")
ax[1, 0].plot(t, x_cal * 1e3, lw=0.8, alpha=0.8, label="multipath, calibrated")
ax[1, 0].set_xlim(0, 12)
ax[1, 0].set_title("Recovered chest displacement")
ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("displacement (mm)")
ax[1, 0].legend(fontsize=8)

# (4) Spectrum -- shows the interferer leaking in at its own frequency
f_t, m_t = spectrum(x_true, FS)
f_c, m_c = spectrum(x_cal, FS)
ax[1, 1].plot(f_t, m_t * 1e3, 'k', lw=2, label="ground truth")
ax[1, 1].plot(f_c, m_c * 1e3, lw=1, label="calibrated estimate")
ax[1, 1].axvline(0.30, color='g', ls='--', lw=0.8)
ax[1, 1].axvline(1.20, color='b', ls='--', lw=0.8)
ax[1, 1].axvline(0.75, color='r', ls=':', lw=1.0, label="interferer 0.75 Hz")
ax[1, 1].set_xlim(0, 2.0)
ax[1, 1].set_title("Displacement spectrum (resp 0.30, heart 1.20 Hz)")
ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_ylabel("amplitude (mm)")
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("sixport_demo.png", dpi=130)
print("\nsaved figure -> sixport_demo.png")

# ----- Export the dataset (this is your collected experiment) ---------------
SixPortRadarSimulator.save(data_mp, "sixport_dataset.npz", "sixport_dataset.csv")
print("saved dataset -> sixport_dataset.csv / sixport_dataset.npz")
