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
    SixPortRadarSimulator,
    hardware_from_config,
    simulator_from_yaml,
    extract_iq, calibrate_iq, recover_displacement, spectrum,
)

CONFIG_PATH = "configuration.yaml"

# ----- Scene B: target + configured clutter/interferers ---------------------
sim_mp, DUR, config = simulator_from_yaml(CONFIG_PATH)
FS = sim_mp.fs_hz

# ----- Scene A: clean (main target only) -- the reference -------------------
evaluation = config.get("evaluation", {})
target_path_name = str(evaluation.get("target_path", sim_mp.paths[0].name))
target_path = next((p for p in sim_mp.paths if p.name == target_path_name), sim_mp.paths[0])
sim_clean = SixPortRadarSimulator(
    f0_hz=sim_mp.f0_hz,
    fs_hz=sim_mp.fs_hz,
    direct_path=sim_mp.direct_path,
    hw=hardware_from_config(config.get("hardware")),
)
sim_clean.add_path(target_path)

data_clean = sim_clean.collect(DUR)
data_mp = sim_mp.collect(DUR)
V_clean = data_clean.get("V_external", data_clean["V"])
V_mp = data_mp.get("V_external", data_mp["V"])
lam = sim_mp.wavelength_m

# ----- Ground-truth chest displacement (for error evaluation) ---------------
t = data_mp["t"]
x_true = target_path.motion(t)
x_true = x_true - np.mean(x_true)

# ----- Process CLEAN scene --------------------------------------------------
I0, Q0 = extract_iq(V_clean)
I0c, Q0c, _ = calibrate_iq(I0, Q0)
x_clean = recover_displacement(I0c, Q0c, lam)

# ----- Process MULTIPATH scene ----------------------------------------------
I1, Q1 = extract_iq(V_mp)
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
    ax[0, 0].plot(t[:400], V_mp[i][:400], lw=0.9, label=f"v{i+1}")
ax[0, 0].set_title("Four-detector voltages after direct-path subtraction")
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
motion_config = next(
    (p.get("motion", {}) for p in config.get("paths", [])
     if p.get("name") == target_path.name),
    {},
)
if "resp_hz" in motion_config:
    ax[1, 1].axvline(float(motion_config["resp_hz"]), color='g', ls='--', lw=0.8)
if "heart_hz" in motion_config:
    ax[1, 1].axvline(float(motion_config["heart_hz"]), color='b', ls='--', lw=0.8)
for path_config in config.get("paths", []):
    motion = path_config.get("motion", {})
    if motion.get("type") == "sinusoid" and "freq_hz" in motion:
        freq = float(motion["freq_hz"])
        label = f"{path_config.get('name', 'path')} {freq:g} Hz"
        ax[1, 1].axvline(freq, color='r', ls=':', lw=1.0, label=label)
ax[1, 1].set_xlim(0, 2.0)
ax[1, 1].set_title("Displacement spectrum")
ax[1, 1].set_xlabel("Hz"); ax[1, 1].set_ylabel("amplitude (mm)")
ax[1, 1].legend(fontsize=8)

fig.tight_layout()
output_config = config.get("outputs", {})
figure_path = str(output_config.get("figure_png", "sixport_demo.png"))
npz_path = str(output_config.get("dataset_npz", "sixport_dataset.npz"))
csv_path = str(output_config.get("dataset_csv", "sixport_dataset.csv"))
fig.savefig(figure_path, dpi=130)
print(f"\nsaved figure -> {figure_path}")

# ----- Export the dataset (this is your collected experiment) ---------------
SixPortRadarSimulator.save(data_mp, npz_path, csv_path)
print(f"saved dataset -> {csv_path} / {npz_path}")
