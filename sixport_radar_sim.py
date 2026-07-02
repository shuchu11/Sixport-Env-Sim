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
from pathlib import Path as FilePath
from typing import Any, Callable, List, Optional, Tuple

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
# Configuration loading
# ---------------------------------------------------------------------------
def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value[0:1] in ("'", '"') and value[-1:] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part.strip()) for part in body.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _prepare_yaml_lines(text: str) -> List[Tuple[int, str]]:
    lines = []
    for raw in text.splitlines():
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))
    return lines


def _parse_yaml_block(lines: List[Tuple[int, str]], start: int, indent: int) -> Tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    is_list = lines[start][0] == indent and lines[start][1].startswith("- ")
    if is_list:
        values = []
        idx = start
        while idx < len(lines):
            line_indent, text = lines[idx]
            if line_indent != indent or not text.startswith("- "):
                break
            item_text = text[2:].strip()
            idx += 1
            if item_text == "":
                item, idx = _parse_yaml_block(lines, idx, indent + 2)
                values.append(item)
                continue
            if ":" in item_text:
                key, raw_value = item_text.split(":", 1)
                item = {}
                raw_value = raw_value.strip()
                if raw_value:
                    item[key.strip()] = _parse_scalar(raw_value)
                else:
                    item[key.strip()], idx = _parse_yaml_block(lines, idx, indent + 2)
                if idx < len(lines) and lines[idx][0] > indent:
                    extra, idx = _parse_yaml_block(lines, idx, indent + 2)
                    if isinstance(extra, dict):
                        item.update(extra)
                values.append(item)
            else:
                values.append(_parse_scalar(item_text))
        return values, idx

    values = {}
    idx = start
    while idx < len(lines):
        line_indent, text = lines[idx]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation near: {text}")
        if ":" not in text:
            raise ValueError(f"Expected 'key: value' near: {text}")
        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        idx += 1
        if raw_value:
            values[key] = _parse_scalar(raw_value)
        else:
            values[key], idx = _parse_yaml_block(lines, idx, indent + 2)
    return values, idx


def load_yaml_config(path: str) -> dict:
    """Load the small YAML subset used by the simulator configuration files."""
    text = FilePath(path).read_text(encoding="utf-8")
    lines = _prepare_yaml_lines(text)
    if not lines:
        return {}
    config, idx = _parse_yaml_block(lines, 0, lines[0][0])
    if idx != len(lines):
        raise ValueError("Could not parse the full YAML configuration.")
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML configuration must be a mapping.")
    return config


def motion_from_config(config: Optional[dict]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a motion model from a YAML motion block."""
    if config is None:
        return static_motion()
    motion_type = str(config.get("type", "static")).lower()
    if motion_type == "static":
        return static_motion()
    if motion_type == "sinusoid":
        return sinusoid_motion(
            amplitude_m=float(config["amplitude_m"]),
            freq_hz=float(config["freq_hz"]),
            phase_rad=float(config.get("phase_rad", 0.0)),
        )
    if motion_type == "vital_signs":
        return vital_signs_motion(
            resp_amp_m=float(config.get("resp_amp_m", 4e-3)),
            resp_hz=float(config.get("resp_hz", 0.30)),
            heart_amp_m=float(config.get("heart_amp_m", 0.4e-3)),
            heart_hz=float(config.get("heart_hz", 1.20)),
            resp_phase=float(config.get("resp_phase", 0.0)),
            heart_phase=float(config.get("heart_phase", 0.0)),
        )
    if motion_type == "random_walk":
        return random_walk_motion(
            step_std_m=float(config["step_std_m"]),
            seed=config.get("seed"),
        )
    raise ValueError(f"Unknown motion type: {motion_type}")


def hardware_from_config(config: Optional[dict]) -> SixPortHardware:
    """Create a SixPortHardware instance from a YAML hardware block."""
    if config is None:
        return SixPortHardware()
    return SixPortHardware(
        lo_amplitude=float(config.get("lo_amplitude", 1.0)),
        phase_err_deg=tuple(config.get("phase_err_deg", (0.0, 0.0, 0.0, 0.0))),
        responsivity=tuple(config.get("responsivity", (1.0, 1.0, 1.0, 1.0))),
        nonlinearity=tuple(config.get("nonlinearity", (0.0, 0.0, 0.0, 0.0))),
        dc_offset=tuple(config.get("dc_offset", (0.0, 0.0, 0.0, 0.0))),
        noise_v_rms=float(config.get("noise_v_rms", 0.0)),
        seed=config.get("seed"),
    )


def path_from_config(config: dict) -> Path:
    """Create one propagation path from a YAML path block."""
    return Path(
        distance_m=float(config["distance_m"]),
        reflectivity=float(config.get("reflectivity", 1.0)),
        motion=motion_from_config(config.get("motion")),
        extra_phase=float(config.get("extra_phase", 0.0)),
        name=str(config.get("name", "path")),
    )


def direct_path_from_config(config: Optional[dict]) -> Optional[DirectPath]:
    """Create the direct Tx-to-Rx leakage path from a YAML block."""
    if not config or not config.get("enabled", False):
        return None
    return DirectPath(
        distance_m=float(config.get("distance_m", 0.15)),
        transmission_loss_db=float(config.get("transmission_loss_db", 30.0)),
        extra_phase=float(config.get("extra_phase", 0.0)),
        subtract_voltage_baseline=bool(config.get("subtract_voltage_baseline", True)),
        name=str(config.get("name", "tx_rx_direct")),
    )


def simulator_from_config(config: dict) -> Tuple[SixPortRadarSimulator, float]:
    """Create a simulator and duration from a parsed YAML configuration."""
    radar = config.get("radar", {})
    sim = SixPortRadarSimulator(
        f0_hz=float(radar.get("f0_hz", 24e9)),
        fs_hz=float(radar.get("fs_hz", 200.0)),
        direct_path=direct_path_from_config(config.get("direct_path")),
        hw=hardware_from_config(config.get("hardware")),
    )
    for path_config in config.get("paths", []):
        sim.add_path(path_from_config(path_config))
    duration_s = float(radar.get("duration_s", 30.0))
    return sim, duration_s


def simulator_from_yaml(path: str) -> Tuple[SixPortRadarSimulator, float, dict]:
    """Load configuration.yaml and return (simulator, duration_s, config)."""
    config = load_yaml_config(path)
    sim, duration_s = simulator_from_config(config)
    return sim, duration_s, config


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


@dataclass
class DirectPath:
    """
    Direct Tx-to-Rx leakage/coupling path.

    This is not an external reflection. It is a one-way propagation/coupling
    term, so its phase uses 2*pi*d/lambda instead of the reflected-path
    4*pi*d/lambda.
    """
    distance_m: float = 0.15
    transmission_loss_db: float = 30.0
    extra_phase: float = 0.0
    subtract_voltage_baseline: bool = True
    name: str = "tx_rx_direct"

    @property
    def amplitude(self) -> float:
        return 10.0 ** (-self.transmission_loss_db / 20.0)

    def complex_return(self, t: np.ndarray, wavelength_m: float) -> np.ndarray:
        phi = (2.0 * np.pi / wavelength_m) * self.distance_m + self.extra_phase
        value = self.amplitude * np.exp(1j * phi)
        return np.full(len(t), value, dtype=complex)


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

    A and B are complex envelopes/phasors at the carrier, not sampled
    passband sinusoids. The common exp(j*2*pi*f0*t) carrier is factored out.

    Imperfections you can sweep for your research:
      lo_amplitude        : reference CW complex-envelope magnitude |A|
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

    def detect(self, B: np.ndarray, A: Optional[np.ndarray | complex | float] = None,
               add_noise: bool = True) -> np.ndarray:
        """
        Given received complex envelope B(t) (shape [N]), produce the four
        detector voltages (shape [4, N]) -- the raw "experimental data".

        If A is omitted, a stable CW reference is represented by a constant
        complex envelope with magnitude lo_amplitude. The actual RF waveform
        would be Re{A * exp(j*2*pi*f0*t)} before carrier factoring.
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
            if add_noise and self.noise_v_rms > 0:
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
    direct_path : optional direct Tx-to-Rx leakage/coupling term
    hw      : SixPortHardware model
    """
    f0_hz: float = 24.0e9
    fs_hz: float = 1000.0
    paths: List[Path] = field(default_factory=list)
    direct_path: Optional[DirectPath] = None
    hw: SixPortHardware = field(default_factory=SixPortHardware)

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.f0_hz

    def add_path(self, path: Path) -> "SixPortRadarSimulator":
        self.paths.append(path)
        return self

    def baseband(self, t: np.ndarray) -> np.ndarray:
        """External reflected return B_ext(t) = sum over target/clutter paths."""
        B = np.zeros(len(t), dtype=complex)
        for p in self.paths:
            B += p.complex_return(t, self.wavelength_m)
        return B

    def direct_baseband(self, t: np.ndarray) -> np.ndarray:
        """Direct Tx-to-Rx leakage term, if configured."""
        if self.direct_path is None:
            return np.zeros(len(t), dtype=complex)
        return self.direct_path.complex_return(t, self.wavelength_m)

    def collect(self, duration_s: float) -> dict:
        """
        Run an 'experiment': returns a dict with
          t            : time vector
          V            : (4, N) raw detector voltages, including direct leakage
          V_direct     : (4, N) detector baseline from the direct path alone
          V_external   : (4, N) V - V_direct, if direct subtraction is enabled
          B_true       : complex baseband ground truth, external + direct paths
          B_external   : complex baseband from external reflected paths only
          B_direct     : complex baseband from direct Tx-to-Rx coupling only
          per_path     : dict name -> complex baseband of that path alone
          meta         : configuration metadata
        """
        n = int(round(duration_s * self.fs_hz))
        t = np.arange(n) / self.fs_hz
        B_external = self.baseband(t)
        B_direct = self.direct_baseband(t)
        B = B_external + B_direct
        V = self.hw.detect(B)
        V_direct = self.hw.detect(B_direct, add_noise=False)
        V_external = None
        if self.direct_path is not None and self.direct_path.subtract_voltage_baseline:
            V_external = V - V_direct
        per_path = {p.name: p.complex_return(t, self.wavelength_m) for p in self.paths}
        if self.direct_path is not None:
            per_path[self.direct_path.name] = B_direct
        meta = {
            "f0_hz": self.f0_hz,
            "fs_hz": self.fs_hz,
            "wavelength_m": self.wavelength_m,
            "duration_s": duration_s,
            "n_samples": n,
            "direct_path": None if self.direct_path is None else {
                "name": self.direct_path.name,
                "distance_m": self.direct_path.distance_m,
                "transmission_loss_db": self.direct_path.transmission_loss_db,
                "amplitude": self.direct_path.amplitude,
                "subtract_voltage_baseline": self.direct_path.subtract_voltage_baseline,
            },
            "paths": [{"name": p.name, "distance_m": p.distance_m,
                       "reflectivity": p.reflectivity} for p in self.paths],
        }
        data = {
            "t": t,
            "V": V,
            "V_direct": V_direct,
            "B_true": B,
            "B_external": B_external,
            "B_direct": B_direct,
            "per_path": per_path,
            "meta": meta,
        }
        if V_external is not None:
            data["V_external"] = V_external
        return data

    # ---- persistence: this is your "data collection" output -------------
    @staticmethod
    def save(data: dict, npz_path: str, csv_path: Optional[str] = None) -> None:
        zero_complex = np.zeros_like(data["B_true"])
        arrays = {
            "t": data["t"],
            "V": data["V"],
            "V_direct": data["V_direct"],
            "B_true_real": data["B_true"].real,
            "B_true_imag": data["B_true"].imag,
            "B_external_real": data.get("B_external", data["B_true"]).real,
            "B_external_imag": data.get("B_external", data["B_true"]).imag,
            "B_direct_real": data.get("B_direct", zero_complex).real,
            "B_direct_imag": data.get("B_direct", zero_complex).imag,
            "meta": json.dumps(data["meta"]),
        }
        if "V_external" in data:
            arrays["V_external"] = data["V_external"]
        np.savez_compressed(npz_path, **arrays)
        if csv_path:
            V = data["V"]
            header = "t,v1,v2,v3,v4"
            columns = [data["t"], V[0], V[1], V[2], V[3]]
            if "V_external" in data:
                Ve = data["V_external"]
                columns.extend([Ve[0], Ve[1], Ve[2], Ve[3]])
                header += ",v1_external,v2_external,v3_external,v4_external"
            arr = np.column_stack(columns)
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
