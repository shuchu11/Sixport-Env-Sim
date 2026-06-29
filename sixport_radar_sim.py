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
# 1. Motion models  (each returns displacement x(t) in metres)  : 目標物震動行為
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


def random_walk_motion(step_std_m: float,
                       seed: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """Brownian displacement, e.g. for a slowly drifting interferer."""
    rng = np.random.default_rng(seed)

    def x(t):
        n = len(t)
        steps = rng.normal(0.0, step_std_m, size=n)
        return np.cumsum(steps)
    return x


# ---------------------------------------------------------------------------
# 2. Path / target description       反射回來的波函數
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
# 3. Six-port hardware model (imperfections live here)   量測4-port功率，加入硬體干擾選項。目前輸入波振幅是定值感覺有誤
# ---------------------------------------------------------------------------
@dataclass
class SixPortHardware:
    """
    Models the four-detector six-port quadrature demodulator.

    Ideal detector powers follow the four six-port node equations:
        v1 = 1/4 | -A + jB |^2
        v2 = 1/4 | jA - B  |^2
        v3 = 1/4 | -A + B  |^2
        v4 = 1/4 | jA + jB |^2

    A is the reference/input wave and B is the received/output wave.

    Imperfections you can sweep for your research:
      lo_amplitude        : reference/input wave magnitude |A|
      phase_err_deg       : per-detector phase error added to the B branch
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
        self._phase_err = np.deg2rad(np.array(self.phase_err_deg))
        self._rng = np.random.default_rng(self.seed)

    def detect(self, B: np.ndarray, A: Optional[np.ndarray | complex | float] = None) -> np.ndarray:
        """
        Given complex wave B(t) (shape [N]) produce the four detector voltages
        (shape [4, N]) -- the raw "experimental data".

        If A is omitted, a constant reference/input wave with magnitude
        lo_amplitude is used.
        """
        if A is None:
            A = self.lo_amplitude

        A = np.asarray(A, dtype=complex)
        N = len(B)
        V = np.empty((4, N), dtype=float)

        a_coeff = np.array([-1.0, 1.0j, -1.0, 1.0j], dtype=complex)
        b_coeff = np.array([1.0j, -1.0, 1.0, 1.0j], dtype=complex)
        b_coeff = b_coeff * np.exp(1j * self._phase_err)

        for i in range(4):
            # Six-port detector node power from the ideal equations above.
            P = 0.25 * np.abs(a_coeff[i] * A + b_coeff[i] * B) ** 2
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
        I = v4 - v3 = Re{A B*}
        Q = v2 - v1 = Im{A B*}
    This cancels the common 1/4 * (|A|^2 + |B|^2) terms in the ideal case.
    """
    I = V[3] - V[2]
    Q = V[1] - V[0]
    return I, Q


def calibrate_iq(I: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Ellipse-based calibration:
      1) remove DC offset (ellipse center)  -> kills STATIC multipath term
      2) correct amplitude + phase imbalance -> turn ellipse back into a circle
    Returns calibrated (I_c, Q_c) plus the fit dict.
    """
    I = np.asarray(I, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    if I.size != Q.size or I.size < 6:
        raise ValueError("I and Q must have the same length and at least 6 samples.")

    # Match the normalized conic used in cw_iq_imbalance.py:
    #     I^2 + A*Q^2 + B*I*Q + C*I + D*Q + E = 0
    # This parameterization gives the amplitude and phase imbalance directly.
    M = np.column_stack((Q ** 2, I * Q, I, Q, np.ones_like(I)))
    A, B, C, D, E = np.linalg.lstsq(M, -(I ** 2), rcond=None)[0]
    if A <= 0 or B * B - 4.0 * A >= 0:
        raise ValueError("The fitted conic is not a valid ellipse.")

    hessian = np.array([[2.0, B], [B, 2.0 * A]])
    x0, y0 = np.linalg.solve(hessian, -np.array([C, D]))
    xc = I - x0
    yc = Q - y0

    amp_imbalance = np.sqrt(1.0 / A)
    phi = np.arcsin(np.clip(B / (2.0 * np.sqrt(A)), -1.0, 1.0))

    def correct_with_phase(phase: float) -> Tuple[np.ndarray, np.ndarray, float]:
        if abs(np.cos(phase)) < 1e-8:
            raise ValueError("Phase imbalance is too close to +/-90 degrees.")
        q_corr = (yc / amp_imbalance - xc * np.sin(phase)) / np.cos(phase)
        radius = np.hypot(xc, q_corr)
        radial_cv = np.std(radius) / (np.mean(radius) + 1e-12)
        return xc, q_corr, float(radial_cv)

    # Six-port I/Q extraction uses angle(A*conj(B)); depending on channel signs,
    # Eq. (5)'s phase sign can be reversed. Choose the more circular result.
    candidates = [(phi, correct_with_phase(phi))]
    if abs(phi) > 1e-12:
        candidates.append((-phi, correct_with_phase(-phi)))
    phase_used, (I_c, Q_c, radial_cv) = min(candidates, key=lambda item: item[1][2])

    fit = {
        "coef": (1.0, B, A, C, D, E),
        "center": (float(x0), float(y0)),
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "E": float(E),
        "amplitude_imbalance": float(amp_imbalance),
        "phase_imbalance_rad": float(phi),
        "phase_imbalance_deg": float(np.rad2deg(phi)),
        "phase_used_rad": float(phase_used),
        "phase_used_deg": float(np.rad2deg(phase_used)),
        "radial_cv": radial_cv,
    }
    return I_c, Q_c, fit


def recover_displacement(I: np.ndarray, Q: np.ndarray, wavelength_m: float
                         ) -> np.ndarray:
    """
    Phase demodulation -> displacement.
    The four-detector equations above recover A*B-conjugate phase when A is
    the constant reference wave:
        atan2(Q, I) = angle(A B*) = -angle(B) + constant.
    Since target displacement increases angle(B) by 4*pi*x/lambda, the
    recovered displacement needs the negative sign below.

    phi(t) = atan2(Q, I); unwrap; x(t) = -phi * lambda / (4*pi).
    Mean-subtracted (only relative displacement is observable in CW).
    """
    phi = np.unwrap(np.arctan2(Q, I))
    x = -phi * wavelength_m / (4.0 * np.pi)
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
