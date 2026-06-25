# -*- coding: utf-8 -*-
"""
sixport_radar_sim.py
====================
A configurable six-port CW (continuous-wave) radar simulator built for
research on MULTIPATH / multi-target interference mitigation.

Why this design
---------------
A six-port front-end performs quadrature (I/Q) demodulation using only passive
microwave components (couplers / power dividers / hybrids) plus four power
detectors. The "experimental data" a real six-port radar produces is the set
of four detector voltages over time:  v1(t), v2(t), v3(t), v4(t).

This simulator reproduces that data layer faithfully, including the
imperfections that make six-port radar interesting:
  * detector responsivity mismatch (gain imbalance)
  * detector square-law non-linearity (2nd-order term)
  * port phase errors (the four reference phases are not exactly 0/90/180/270)
  * LO-RF leakage / DC offset
  * additive noise

Crucially, the RF input is the SUM of an arbitrary number of propagation paths.
That is exactly how multipath / multi-target interference enters a real system:

      B(t) = sum_k  rho_k * exp( j * phi_k(t) )

  - a main target (e.g. chest wall: respiration + heartbeat)
  - static clutter / multipath (a wall, a table -> constant phasor -> DC offset)
  - a second moving target / dynamic multipath (a fan, another person)

You configure the scene, "collect" the four-detector data, and then test your
own mitigation pipeline against the provided baseline (ellipse calibration +
arctan-demodulation).

Units: SI. Distances in metres, time in seconds, frequency in Hz.

Author: built for radar research use. MIT-style: do whatever you like with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Tuple

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


# ---------------------------------------------------------------------------
# 1. Motion models  (each returns displacement x(t) in metres)
# ---------------------------------------------------------------------------
def static_motion() -> Callable[[np.ndarray], np.ndarray]:
    """No motion. Useful for clutter / static multipath."""
    return lambda t: np.zeros_like(t)


def sinusoid_motion(amplitude_m: float, freq_hz: float,
                    phase_rad: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """Pure sinusoidal vibration. Good for a vibrating-plate test target."""
    return lambda t: amplitude_m * np.sin(2 * np.pi * freq_hz * t + phase_rad)


def vital_signs_motion(resp_amp_m: float = 4e-3, resp_hz: float = 0.30,
                       heart_amp_m: float = 0.4e-3, heart_hz: float = 1.2,
                       resp_phase: float = 0.0,
                       heart_phase: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Chest-wall displacement = respiration (large, slow) + heartbeat (small, fast).
    Typical orders of magnitude: respiration ~1-6 mm @ 0.2-0.5 Hz,
    heartbeat ~0.1-0.5 mm @ ~1-1.5 Hz.
    """
    def x(t):
        return (resp_amp_m * np.sin(2 * np.pi * resp_hz * t + resp_phase)
                + heart_amp_m * np.sin(2 * np.pi * heart_hz * t + heart_phase))
    return x


def random_walk_motion(step_std_m: float, fs: float,
                       seed: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """Brownian displacement, e.g. for a slowly drifting interferer."""
    rng = np.random.default_rng(seed)

    def x(t):
        n = len(t)
        steps = rng.normal(0.0, step_std_m, size=n)
        return np.cumsum(steps)
    return x


# ---------------------------------------------------------------------------
# 2. Path / target description
# ---------------------------------------------------------------------------
@dataclass
class Path:
    """
    One propagation path between radar and a (possibly moving) reflector.

    distance_m    : nominal one-way distance d_k (round trip = 2*d_k)
    reflectivity  : |rho_k|, lumps RCS + path loss + antenna gains (linear, 0..1+)
    motion        : callable t -> x(t) displacement in metres (superposed on d_k)
    extra_phase   : fixed phase offset (rad), e.g. reflection phase of the surface
    name          : label for bookkeeping / plotting
    """
    distance_m: float
    reflectivity: float
    motion: Callable[[np.ndarray], np.ndarray] = field(default_factory=static_motion)
    extra_phase: float = 0.0
    name: str = "path"

    def complex_return(self, t: np.ndarray, wavelength_m: float) -> np.ndarray:
        """Baseband complex contribution rho_k * exp(j*phi_k(t)) (monostatic)."""
        x = self.motion(t)
        # round-trip phase: 2 * (2*pi/lambda) * (d + x) = (4*pi/lambda)*(d+x)
        phi = (4.0 * np.pi / wavelength_m) * (self.distance_m + x) + self.extra_phase
        return self.reflectivity * np.exp(1j * phi)


# ---------------------------------------------------------------------------
# 3. Six-port hardware model (imperfections live here)
# ---------------------------------------------------------------------------
@dataclass
class SixPortHardware:
    """
    Models the four-detector six-port quadrature demodulator.

    Ideal: detector i measures power of (RF + LO*exp(j*theta_i)),
    theta_i = [0, 90, 180, 270] deg. Differential combination -> I, Q.

    Imperfections you can sweep for your research:
      lo_amplitude        : reference (LO) magnitude L
      phase_err_deg       : per-detector phase error added to the ideal theta_i
      responsivity        : per-detector power->voltage gain (mismatch = imbalance)
      nonlinearity        : per-detector 2nd-order term gamma_i (V = R*P + gamma*P^2)
      dc_offset           : per-detector additive DC (LO self-mixing / bias)
      noise_v_rms         : additive Gaussian voltage noise at each detector
    """
    lo_amplitude: float = 1.0
    phase_err_deg: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    responsivity: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    nonlinearity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    dc_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    noise_v_rms: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        self._theta = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0])
                                 + np.array(self.phase_err_deg))
        self._rng = np.random.default_rng(self.seed)

    def detect(self, B: np.ndarray) -> np.ndarray:
        """
        Given complex baseband return B(t) (shape [N]) produce the four
        detector voltages (shape [4, N]) -- the raw "experimental data".
        """
        L = self.lo_amplitude
        N = len(B)
        V = np.empty((4, N), dtype=float)
        for i in range(4):
            # power at detector i: |B + L*exp(j*theta_i)|^2
            ref = L * np.exp(1j * self._theta[i])
            P = np.abs(B + ref) ** 2
            R = self.responsivity[i]
            g = self.nonlinearity[i]
            Vi = R * P + g * P ** 2 + self.dc_offset[i]
            if self.noise_v_rms > 0:
                Vi = Vi + self._rng.normal(0.0, self.noise_v_rms, size=N)
            V[i] = Vi
        return V


# ---------------------------------------------------------------------------
# 4. The simulator
# ---------------------------------------------------------------------------
@dataclass
class SixPortRadarSimulator:
    """
    Top-level scene + acquisition.

    f0_hz   : carrier frequency (e.g. 24e9 for K-band, 5.8e9 for ISM, 60e9 mmWave)
    fs_hz   : sampling rate of the four baseband detector channels
    paths   : list of Path objects (main target + clutter + interferers)
    hw      : SixPortHardware model
    """
    f0_hz: float = 24.0e9
    fs_hz: float = 1000.0
    paths: List[Path] = field(default_factory=list)
    hw: SixPortHardware = field(default_factory=SixPortHardware)

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.f0_hz

    def add_path(self, path: Path) -> "SixPortRadarSimulator":
        self.paths.append(path)
        return self

    def baseband(self, t: np.ndarray) -> np.ndarray:
        """Total complex return B(t) = sum over paths. This is the ground truth."""
        B = np.zeros(len(t), dtype=complex)
        for p in self.paths:
            B += p.complex_return(t, self.wavelength_m)
        return B

    def collect(self, duration_s: float) -> dict:
        """
        Run an 'experiment': returns a dict with
          t            : time vector
          V            : (4, N) raw detector voltages  <-- your measured data
          B_true       : complex baseband ground truth (all paths summed)
          per_path     : dict name -> complex baseband of that path alone
          meta         : configuration metadata
        """
        n = int(round(duration_s * self.fs_hz))
        t = np.arange(n) / self.fs_hz
        B = self.baseband(t)
        V = self.hw.detect(B)
        per_path = {p.name: p.complex_return(t, self.wavelength_m) for p in self.paths}
        meta = {
            "f0_hz": self.f0_hz,
            "fs_hz": self.fs_hz,
            "wavelength_m": self.wavelength_m,
            "duration_s": duration_s,
            "n_samples": n,
            "paths": [{"name": p.name, "distance_m": p.distance_m,
                       "reflectivity": p.reflectivity} for p in self.paths],
        }
        return {"t": t, "V": V, "B_true": B, "per_path": per_path, "meta": meta}

    # ---- persistence: this is your "data collection" output -------------
    @staticmethod
    def save(data: dict, npz_path: str, csv_path: Optional[str] = None) -> None:
        np.savez_compressed(
            npz_path,
            t=data["t"], V=data["V"],
            B_true_real=data["B_true"].real, B_true_imag=data["B_true"].imag,
            meta=json.dumps(data["meta"]),
        )
        if csv_path:
            V = data["V"]
            arr = np.column_stack([data["t"], V[0], V[1], V[2], V[3]])
            header = "t,v1,v2,v3,v4"
            np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")


# ---------------------------------------------------------------------------
# 5. Baseline processing pipeline (the thing your method must beat)
# ---------------------------------------------------------------------------
def extract_iq(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differential six-port combination:
        I = v1 - v3   (theta 0 vs 180)
        Q = v2 - v4   (theta 90 vs 270)
    This cancels the common |B|^2 + L^2 terms in the ideal case.
    """
    I = V[0] - V[2]
    Q = V[1] - V[3]
    return I, Q


def fit_ellipse(I: np.ndarray, Q: np.ndarray) -> dict:
    """
    Algebraic conic fit  a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0
    (Fitzgibbon-style, least squares). Returns geometric ellipse parameters
    used to undo the six-port I/Q imbalance + DC offset (incl. STATIC multipath).
    """
    x = I.astype(float)
    y = Q.astype(float)
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    # Solve the homogeneous system via SVD (smallest singular vector).
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    a, b, c, d, e, f = Vt[-1]

    # Center of ellipse
    denom = b * b - 4 * a * c
    if abs(denom) < 1e-20:
        denom = 1e-20
    x0 = (2 * c * d - b * e) / denom
    y0 = (2 * a * e - b * d) / denom
    return {"coef": (a, b, c, d, e, f), "center": (x0, y0)}


def calibrate_iq(I: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Ellipse-based calibration:
      1) remove DC offset (ellipse center)  -> kills STATIC multipath term
      2) correct amplitude + phase imbalance -> turn ellipse back into a circle
    Returns calibrated (I_c, Q_c) plus the fit dict.
    """
    fit = fit_ellipse(I, Q)
    x0, y0 = fit["center"]
    xc = I - x0
    yc = Q - y0

    # Gram-Schmidt / amplitude-phase imbalance correction from the conic coefs.
    a, b, c, d, e, f = fit["coef"]
    # rotation angle of the ellipse
    if abs(a - c) < 1e-20:
        theta = np.pi / 4 if b > 0 else -np.pi / 4
    else:
        theta = 0.5 * np.arctan2(b, (a - c))
    ct, st = np.cos(theta), np.sin(theta)
    # rotate into ellipse principal axes
    xr = ct * xc + st * yc
    yr = -st * xc + ct * yc
    # estimate semi-axes from the rotated point cloud (robust to noise)
    ax = np.sqrt(np.mean(xr ** 2)) + 1e-12
    ay = np.sqrt(np.mean(yr ** 2)) + 1e-12
    scale = (ax + ay) / 2.0
    xr *= scale / ax
    yr *= scale / ay
    # rotate back
    I_c = ct * xr - st * yr
    Q_c = st * xr + ct * yr
    fit["theta"] = theta
    fit["semi_axes"] = (ax, ay)
    return I_c, Q_c, fit


def recover_displacement(I: np.ndarray, Q: np.ndarray, wavelength_m: float
                         ) -> np.ndarray:
    """
    Phase demodulation -> displacement.
    phi(t) = atan2(Q, I); unwrap; x(t) = phi * lambda / (4*pi).
    Mean-subtracted (only relative displacement is observable in CW).
    """
    phi = np.unwrap(np.arctan2(Q, I))
    x = phi * wavelength_m / (4.0 * np.pi)
    return x - np.mean(x)


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum, handy for separating motion frequencies."""
    n = len(x)
    w = np.hanning(n)
    X = np.fft.rfft((x - np.mean(x)) * w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X) * 2.0 / np.sum(w)
    return f, mag


# ---------------------------------------------------------------------------
# 6. Convenience scene builders
# ---------------------------------------------------------------------------
def scene_vital_signs_with_multipath(
        f0_hz: float = 24e9, fs_hz: float = 200.0,
        target_distance_m: float = 0.5,
        static_clutter: bool = True,
        moving_interferer: bool = True,
        hw: Optional[SixPortHardware] = None) -> SixPortRadarSimulator:
    """
    A ready-made multipath research scene:
      - main target: chest wall (respiration + heartbeat) at target_distance_m
      - optional static multipath: a wall a bit farther, strong, no motion
      - optional dynamic multipath: a fan/second person, different distance & freq
    """
    if hw is None:
        # mildly imperfect hardware (realistic, gives an elliptical raw IQ)
        hw = SixPortHardware(
            lo_amplitude=1.0,
            phase_err_deg=(0.0, 8.0, -5.0, 6.0),
            responsivity=(1.0, 0.85, 1.1, 0.95),
            nonlinearity=(0.02, 0.0, 0.03, 0.01),
            dc_offset=(0.0, 0.0, 0.0, 0.0),
            noise_v_rms=0.002,
            seed=1,
        )
    sim = SixPortRadarSimulator(f0_hz=f0_hz, fs_hz=fs_hz, hw=hw)
    sim.add_path(Path(
        distance_m=target_distance_m, reflectivity=1.0,
        motion=vital_signs_motion(resp_amp_m=4e-3, resp_hz=0.30,
                                  heart_amp_m=0.4e-3, heart_hz=1.20),
        name="chest"))
    if static_clutter:
        sim.add_path(Path(
            distance_m=target_distance_m + 0.37, reflectivity=0.8,
            motion=static_motion(), extra_phase=1.1, name="wall_clutter"))
    if moving_interferer:
        sim.add_path(Path(
            distance_m=target_distance_m + 0.20, reflectivity=0.5,
            motion=sinusoid_motion(amplitude_m=2e-3, freq_hz=0.75),
            name="moving_interferer"))
    return sim
