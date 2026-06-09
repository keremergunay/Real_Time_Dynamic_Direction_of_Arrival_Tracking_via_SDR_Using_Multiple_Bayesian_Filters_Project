"""
BPSK DoA Estimation with Multi-Filter Support (PyQt5 GUI)
==========================================================
AD9361 (ADALM-Pluto) SDR üzerinden BPSK sinyalleri ile gerçek zamanlı
Yön Kestirimi (DoA) ve çoklu Bayesian filtre desteği.

Filters: g-h, 1D Kalman, Particle Filter, IMM Kalman

---------------------------------------------------------------------------
Incremental improvements (no simulation) — "chapters"
  Ch.1: Paths + JSON state (save/load boresight phase, antenna spacing)
  Ch.2: logging to stderr for ops / diagnostics
  Ch.3: SDR finally — disable TX/RX on exit (RF safety)
  Ch.4: Live link metrics (Barker peak/mean, tracked CFO) in GUI
  Ch.5: Optional CSV time-series from GUI thread (raw / filtered / SNR / …)
---------------------------------------------------------------------------
"""

import csv
import json
import logging
import os
import sys
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import adi

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QStackedWidget, QFormLayout, QFrame, QSizePolicy,
)
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ── SDR Settings ──────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _safe_state_suffix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


SAMPLE_RATE = _env_float("KALMAN_SAMPLE_RATE", 2e6)  # 2 Msps, Barker correlation quality
CENTER_FREQ = _env_float("KALMAN_CENTER_FREQ", 1300e6)
TX_IP = os.environ.get("KALMAN_TX_URI", "ip:192.168.3.1")
RX_IP = os.environ.get("KALMAN_RX_URI", "ip:192.168.2.1")
PHASE_SIGN = 1.0 if _env_float("KALMAN_PHASE_SIGN", 1.0) >= 0 else -1.0
MESSAGE = "KALMAN_PROJE"

# Two separate Plutos = independent reference clocks → residual carrier offset (CFO)
# between TX and RX. Correlation peak/mean is lower until CFO is corrected; use a
# lower Barker lock threshold than with a single device.
DUAL_PLUTO_MODE = _env_bool("KALMAN_DUAL_PLUTO_MODE", True)

SPEED_OF_LIGHT = 3e8
WAVELENGTH = SPEED_OF_LIGHT / CENTER_FREQ

# Physical antenna spacing in metres: fixed to lambda/2 for unambiguous two-element DoA.
ANTENNA_SPACING_M = 0.5 * WAVELENGTH
D_SPACING = ANTENNA_SPACING_M

BARKER_CODE = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
SAMPLES_PER_SYMBOL = 32    # more samples/symbol at 2 Msps for better averaging
BUFFER_LEN = 200
ANGLE_MIN_DEG = -45.0
ANGLE_MAX_DEG = 45.0

# Min peak/mean on abs(Barker correlation). Single-radio: ~4–5; dual Pluto ~2.5–3.5.
SYNC_PEAK_RATIO_MIN = 3.0 if DUAL_PLUTO_MODE else 4.5
# EMA on CFO estimate (Hz) for smooth derotation across buffers
DUAL_PLUTO_CFO_EMA_ALPHA = _env_float("KALMAN_CFO_EMA_ALPHA", 0.12)
# Circular EMA applied to inter-channel phase before converting to angle.
PHASE_EMA_ALPHA = _env_float("KALMAN_PHASE_EMA_ALPHA", 0.12)
# Limit accepted DoA movement. This suppresses frame-to-frame phase jitter while
# still allowing a real target to move across the -45..45 degree window.
MAX_ANGLE_RATE_DEGPS = _env_float("KALMAN_MAX_ANGLE_RATE_DEGPS", 45.0)
# Reject raw DoA measurements that jump more than this from the previous one
OUTLIER_THRESHOLD_DEG = _env_float("KALMAN_OUTLIER_THRESHOLD_DEG", 25.0)
# Accept a large jump after several consecutive outliers; otherwise a real moved
# transmitter can leave the tracker stuck on the old angle forever.
OUTLIER_REACQUIRE_COUNT = 5
# Normalized channel coherence below this is too noisy for a reliable phase DoA.
MIN_PHASE_COHERENCE = _env_float("KALMAN_MIN_PHASE_COHERENCE", 0.35)
# Discard this many RX buffers after SDR init to let hardware settle
RX_WARMUP_FRAMES = 5
# RX gain (dB) — fixed manual gain; tune so received signal is not saturated
RX_GAIN_DB = int(_env_float("KALMAN_RX_GAIN_DB", 50))

# Ch.1 — persist boresight phase (and optional antenna spacing) across runs
_SCRIPT_DIR = Path(__file__).resolve().parent
_STATE_SUFFIX = _safe_state_suffix(os.environ.get("KALMAN_STATE_SUFFIX", ""))
STATE_FILE = _SCRIPT_DIR / (
    f"kalman_proje_state_{_STATE_SUFFIX}.json" if _STATE_SUFFIX else "kalman_proje_state.json"
)
CSV_EXPORT_DIR = _SCRIPT_DIR / "csv_exports"

log = logging.getLogger(__name__)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ═══════════════════════════════════════════════════════════════════════
#  FILTER HIERARCHY
# ═══════════════════════════════════════════════════════════════════════

class Filter(ABC):
    """Abstract base class for all angle-tracking filters."""

    @abstractmethod
    def step(self, measurement: float) -> float:
        """Process one measurement (degrees), return filtered angle."""

    @abstractmethod
    def reset(self):
        """Reset internal state to initial conditions."""

    @property
    @abstractmethod
    def variance(self) -> float:
        """Current estimation variance (degrees²)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── g-h Filter ────────────────────────────────────────────────────────

class GHFilter(Filter):
    """
    Simple g-h (alpha-beta) filter.
    g controls position smoothing, h controls velocity smoothing.
    """

    def __init__(self, g: float = 0.4, h: float = 0.1, dt: float = 0.05):
        self._g0 = g
        self._h0 = h
        self.g = g
        self.h = h
        self.dt = dt
        self._x = 0.0
        self._dx = 0.0
        self._var = 25.0  # initial variance estimate

    def step(self, measurement: float) -> float:
        x_pred = self._x + self._dx * self.dt
        residual = measurement - x_pred
        self._x = x_pred + self.g * residual
        self._dx = self._dx + (self.h / self.dt) * residual
        self._var = (1.0 - self.g) ** 2 * self._var + self.g ** 2 * 25.0
        return self._x

    def reset(self):
        self._x = 0.0
        self._dx = 0.0
        self._var = 25.0

    @property
    def variance(self) -> float:
        return self._var


# ── 1-D Kalman Filter ────────────────────────────────────────────────

class Kalman1D(Filter):
    """
    Standard 1-D Kalman filter with constant-velocity state model.
    State: [angle, angular_velocity]ᵀ
    """

    def __init__(self, q_angle: float = 0.5, q_vel: float = 1.0,
                 r_meas: float = 25.0, dt: float = 0.05):
        self._q_angle0 = q_angle
        self._q_vel0 = q_vel
        self._r_meas0 = r_meas
        self.dt = dt
        self._build(q_angle, q_vel, r_meas)

    def _build(self, q_angle, q_vel, r_meas):
        self.x = np.array([[0.0], [0.0]])
        self.P = np.diag([100.0, 25.0])
        self.Q = np.diag([q_angle ** 2, q_vel ** 2])
        self.R = np.array([[r_meas]])
        self.H = np.array([[1.0, 0.0]])

    def step(self, measurement: float) -> float:
        F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        y = np.array([[measurement]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])

    def reset(self):
        self._build(self._q_angle0, self._q_vel0, self._r_meas0)

    @property
    def variance(self) -> float:
        return float(self.P[0, 0])


# ── Particle Filter ──────────────────────────────────────────────────

class ParticleFilter(Filter):
    """
    Bootstrap particle filter for angle tracking.
    Handles non-Gaussian noise distributions.
    """

    def __init__(self, n_particles: int = 500, process_std: float = 2.0,
                 meas_std: float = 5.0):
        self._n0 = n_particles
        self._pstd0 = process_std
        self._mstd0 = meas_std
        self.n_particles = n_particles
        self.process_std = process_std
        self.meas_std = meas_std
        self._init_particles()

    def _init_particles(self):
        self.particles = np.random.uniform(-90, 90, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self._estimate = 0.0
        self._var = 25.0

    def step(self, measurement: float) -> float:
        self.particles += np.random.normal(0, self.process_std, self.n_particles)

        diff = measurement - self.particles
        self.weights = np.exp(-0.5 * (diff / self.meas_std) ** 2)
        w_sum = self.weights.sum()
        if w_sum < 1e-30:
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= w_sum

        self._estimate = np.average(self.particles, weights=self.weights)
        self._var = np.average((self.particles - self._estimate) ** 2,
                               weights=self.weights)

        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(
                self.n_particles, size=self.n_particles, p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

        return self._estimate

    def reset(self):
        self._init_particles()

    @property
    def variance(self) -> float:
        return self._var


# ── IMM Kalman Filter ────────────────────────────────────────────────

class IMMKalman(Filter):
    """
    Interacting Multiple Model Kalman filter (3 models).
    M0: quasi-static, M1: constant velocity, M2: maneuver
    """

    def __init__(self, dt: float = 0.05, R_deg2: float = 64.0, Pi=None):
        self._dt0 = dt
        self._R0 = R_deg2
        self._Pi0 = Pi
        self.dt = dt
        self.R = R_deg2
        if Pi is None:
            self.Pi = np.array([
                [0.92, 0.06, 0.02],
                [0.06, 0.90, 0.04],
                [0.04, 0.10, 0.86],
            ])
        else:
            self.Pi = np.array(Pi, dtype=np.float64)
        self._init_state()

    def _init_state(self):
        x0 = np.array([[0.0], [0.0]])
        P0 = np.diag([100.0, 25.0])
        self.x = [x0.copy() for _ in range(3)]
        self.P = [P0.copy() for _ in range(3)]
        self.mu = np.array([0.34, 0.33, 0.33])
        self._build_models()
        self._var = 100.0

    def _build_models(self):
        dt = self.dt
        a0 = 0.90
        self.F = [
            np.array([[1.0, dt], [0.0, a0]]),
            np.array([[1.0, dt], [0.0, 1.0]]),
            np.array([[1.0, dt], [0.0, 1.0]]),
        ]
        self.Q = [
            np.diag([0.05 ** 2, 0.20 ** 2]),
            np.diag([0.20 ** 2, 0.60 ** 2]),
            np.diag([0.80 ** 2, 2.50 ** 2]),
        ]
        self.H = np.array([[1.0, 0.0]])
        self.I2 = np.eye(2)

    def set_dt(self, dt: float):
        self.dt = _clamp(dt, 0.002, 0.5)
        self._build_models()

    def step(self, measurement: float) -> float:
        z = np.array([[measurement]])

        c = self.mu @ self.Pi
        c = np.maximum(c, 1e-12)
        mu_ij = (self.mu[:, None] * self.Pi) / c[None, :]

        x_mix, P_mix = [], []
        for j in range(3):
            xj = sum(mu_ij[i, j] * self.x[i] for i in range(3))
            Pj = sum(
                mu_ij[i, j] * (self.P[i] + (self.x[i] - xj) @ (self.x[i] - xj).T)
                for i in range(3)
            )
            x_mix.append(xj)
            P_mix.append(Pj)

        mu_pred = c / c.sum()
        likelihood = np.zeros(3)
        x_new, P_new = [], []

        for j in range(3):
            xk = self.F[j] @ x_mix[j]
            Pk = self.F[j] @ P_mix[j] @ self.F[j].T + self.Q[j]

            y = z - self.H @ xk
            S_val = max(float((self.H @ Pk @ self.H.T + self.R)[0, 0]), 1e-9)
            K = (Pk @ self.H.T) / S_val
            xk = xk + K @ y
            Pk = (self.I2 - K @ self.H) @ Pk

            yv = float(y[0, 0])
            likelihood[j] = np.exp(-0.5 * yv * yv / S_val) / np.sqrt(
                2 * np.pi * S_val
            )
            x_new.append(xk)
            P_new.append(Pk)

        mu = mu_pred * likelihood
        mu_sum = mu.sum()
        self.mu = mu / mu_sum if mu_sum > 1e-12 else np.array([0.34, 0.33, 0.33])

        self.x = x_new
        self.P = P_new

        x_hat = sum(self.mu[j] * self.x[j] for j in range(3))
        P_hat = sum(
            self.mu[j] * (self.P[j] + (self.x[j] - x_hat) @ (self.x[j] - x_hat).T)
            for j in range(3)
        )
        self._var = float(P_hat[0, 0])
        return float(x_hat[0, 0])

    def reset(self):
        self.dt = self._dt0
        self.R = self._R0
        self._init_state()

    @property
    def variance(self) -> float:
        return self._var


# ═══════════════════════════════════════════════════════════════════════
#  FILTER REGISTRY — maps display name → (class, default_params)
# ═══════════════════════════════════════════════════════════════════════

FILTER_REGISTRY = {
    "g-h Filter": (GHFilter, {"g": 0.4, "h": 0.1}),
    "1D Kalman": (Kalman1D, {"q_angle": 0.5, "q_vel": 1.0, "r_meas": 25.0}),
    "Particle Filter": (ParticleFilter, {"n_particles": 500, "process_std": 2.0, "meas_std": 5.0}),
    "IMM Kalman": (IMMKalman, {"R_deg2": 64.0}),
}


# ═══════════════════════════════════════════════════════════════════════
#  SDR HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def text_to_bits(text):
    bits = []
    for char in text:
        bits.extend(int(b) for b in bin(ord(char))[2:].zfill(8))
    return np.array(bits)


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(chars)


def correct_frequency_phase(rx_sig, fs):
    fft_sig = np.fft.fft(rx_sig ** 2, 2048)
    fft_freq = np.fft.fftfreq(2048, 1.0 / fs)
    freq_est = fft_freq[np.argmax(np.abs(fft_sig))] / 2.0
    t = np.arange(len(rx_sig)) / fs
    return rx_sig * np.exp(-1j * 2 * np.pi * freq_est * t), freq_est


def estimate_bpsk_residual_cfo_hz(rx: np.ndarray, fs: float) -> float:
    """
    Coarse baseband CFO (Hz) from BPSK^2 spectral line. Needed when TX and RX
    use separate Plutos (no common clock).
    """
    n = len(rx)
    nfft = min(n, 65536)
    nfft = 1 << int(np.floor(np.log2(max(nfft, 1024))))
    nfft = min(nfft, n)
    if nfft < 1024:
        return 0.0
    x = np.asarray(rx[:nfft], dtype=np.complex128)
    sq = np.fft.fft(x * x, nfft)
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)
    k = int(np.argmax(np.abs(sq)))
    return float(freqs[k] / 2.0)


def derotate_common_cfo(rx0: np.ndarray, rx1: np.ndarray, cfo_hz: float, fs: float):
    """Apply same complex rotation to both RX channels (shared RX LO)."""
    t = np.arange(len(rx0), dtype=np.float64) / fs
    ph = np.exp(-1j * 2 * np.pi * cfo_hz * t).astype(np.complex64)
    return rx0 * ph, rx1 * ph


# ═══════════════════════════════════════════════════════════════════════
#  SDR WORKER (QObject with signals)
# ═══════════════════════════════════════════════════════════════════════

class SDRSignals(QObject):
    """Qt signals emitted from the SDR worker thread."""
    new_data = pyqtSignal(
        float, float, float, float, float, float, float
    )  # raw, filt, var, rate, snr, peak/mean, cfo_hz
    message_decoded = pyqtSignal(str, float, float)  # msg, raw, filtered
    status = pyqtSignal(str)


class SDRWorker:
    """
    Runs the SDR TX/RX loop in a background thread.
    Calls active_filter.step() polymorphically for each measurement.
    """

    def __init__(self):
        self.signals = SDRSignals()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._calibration_offset = 0.0
        self._calibration_buf = deque(maxlen=20)
        # Inter-channel phase offset (radians), set during boresight calibration
        self._phase_cal_offset = 0.0
        self._phase_cal_buf = deque(maxlen=50)
        self._is_calibrated = False

        self._active_filter: Filter = IMMKalman()
        self._last_t = time.perf_counter()
        self._prev_filtered = 0.0
        self._prev_raw_deg = 0.0
        self._have_prev_raw_sample = False
        self._have_prev_filtered_sample = False
        self._phase_ema_vec = None
        self._outlier_count = 0
        self._low_coherence_loops = 0
        self._error_count = 0
        self._no_sync_loops = 0
        self._stream_announced = False
        # Physical baseline spacing (m); may be overridden by kalman_proje_state.json
        self._d_spacing = float(ANTENNA_SPACING_M)
        self._load_calibration_state()

    def _load_calibration_state(self) -> None:
        if not STATE_FILE.is_file():
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("version", 0)) != 1:
                return
            saved_freq = data.get("center_freq_hz")
            if saved_freq is not None and abs(float(saved_freq) - CENTER_FREQ) > 1e3:
                log.warning(
                    "Ignoring %s: center_freq %.0f Hz != current %.0f Hz",
                    STATE_FILE,
                    float(saved_freq),
                    CENTER_FREQ,
                )
                return
            saved_sign = data.get("phase_sign")
            if saved_sign is not None and float(saved_sign) != PHASE_SIGN:
                log.warning(
                    "Ignoring %s: phase_sign %.0f != current %.0f",
                    STATE_FILE,
                    float(saved_sign),
                    PHASE_SIGN,
                )
                return
            self._phase_cal_offset = float(data.get("phase_offset_rad", 0.0))
            self._d_spacing = float(ANTENNA_SPACING_M)
            log.info(
                "Loaded %s: phase_offset=%.4f rad, antenna_spacing_m=%.5f",
                STATE_FILE, self._phase_cal_offset, self._d_spacing,
            )
        except Exception as e:
            log.warning("Could not load %s: %s", STATE_FILE, e)

    def _save_calibration_state(self) -> None:
        try:
            with self._state_lock:
                phase_offset = float(self._phase_cal_offset)
                d_spacing = float(self._d_spacing)
            payload = {
                "version": 1,
                "phase_offset_rad": phase_offset,
                "antenna_spacing_m": d_spacing,
                "center_freq_hz": float(CENTER_FREQ),
                "phase_sign": float(PHASE_SIGN),
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            log.info("Saved calibration state to %s", STATE_FILE)
        except Exception as e:
            log.warning("Could not save state: %s", e)

    @property
    def active_filter(self) -> Filter:
        with self._lock:
            return self._active_filter

    @active_filter.setter
    def active_filter(self, f: Filter):
        with self._lock:
            self._active_filter = f
            self._prev_filtered = 0.0
            self._have_prev_filtered_sample = False

    def calibrate(self):
        """Boresight calibration: removes inter-channel phase offset.
        Point the TX at boresight (0°) before pressing calibrate."""
        with self._state_lock:
            buf = list(self._phase_cal_buf)
        if len(buf) > 5:
            # Circular mean of raw phase differences (radians)
            mean_vec = np.mean(np.exp(1j * np.array(buf)))
            phase_offset = float(np.angle(mean_vec))
            with self._state_lock:
                self._phase_cal_offset = phase_offset
                self._is_calibrated = True
            self._phase_ema_vec = None
            offset_deg = np.degrees(phase_offset)
            self.signals.status.emit(
                f"Calibrated — phase offset {offset_deg:.1f}° removed"
            )
            self._save_calibration_state()
        else:
            self.signals.status.emit(
                "No phase snapshots yet — wait until status shows "
                "\"Barker locked\" (TX/RX on, same freq), then calibrate at boresight."
            )

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── main loop ─────────────────────────────────────────────────────

    def _run(self):
        barker_oversampled = np.repeat(BARKER_CODE, SAMPLES_PER_SYMBOL)
        data_bits = text_to_bits(MESSAGE)
        packet_len_samples = (len(data_bits) + len(BARKER_CODE)) * SAMPLES_PER_SYMBOL

        sdr_tx = None
        sdr_rx = None
        try:
            # Ch.3 — TX: safe shutdown via finally
            try:
                self.signals.status.emit("Initializing transmitter…")
                sdr_tx = adi.ad9361(TX_IP)
                sdr_tx.tx_enabled_channels = [0]
                sdr_tx.tx_lo = int(CENTER_FREQ)
                sdr_tx.tx_cyclic_buffer = True
                sdr_tx.sample_rate = int(SAMPLE_RATE)
                sdr_tx.tx_hardwaregain_chan0 = -20

                bpsk_symbols = 2 * data_bits - 1
                tx_frame = np.concatenate(
                    (np.zeros(50), BARKER_CODE, bpsk_symbols, np.zeros(50))
                )
                tx_signal = np.repeat(tx_frame, SAMPLES_PER_SYMBOL) * (2 ** 14)
                sdr_tx.tx(tx_signal.astype(np.complex64))
                self.signals.status.emit("TX ready ✓")
                log.info("Transmitter started on %s at %.3f MHz", TX_IP, CENTER_FREQ / 1e6)
            except Exception as e:
                self.signals.status.emit(f"TX init failed: {e}")
                log.exception("TX init failed")

            try:
                self.signals.status.emit("Initializing receiver…")
                sdr_rx = adi.ad9361(RX_IP)
                sdr_rx.rx_lo = int(CENTER_FREQ)
                sdr_rx.sample_rate = int(SAMPLE_RATE)
                sdr_rx.rx_buffer_size = 65536
                sdr_rx.rx_enabled_channels = [0, 1]
                sdr_rx.gain_control_mode_chan0 = "manual"
                sdr_rx.gain_control_mode_chan1 = "manual"
                sdr_rx.rx_hardwaregain_chan0 = RX_GAIN_DB
                sdr_rx.rx_hardwaregain_chan1 = RX_GAIN_DB
                log.info("Receiver ready on %s, gain %s dB", RX_IP, RX_GAIN_DB)
            except Exception as e:
                self.signals.status.emit(f"RX init failed: {e}")
                log.exception("RX init failed")
                return

            self.signals.status.emit(f"Warming up ({RX_WARMUP_FRAMES} frames)…")
            for _ in range(RX_WARMUP_FRAMES):
                try:
                    sdr_rx.rx()
                except Exception:
                    pass

            self.signals.status.emit("SDR loop running ✓ — waiting for Barker sync…")
            self._last_t = time.perf_counter()
            cfo_filt_hz = 0.0

            while self._running:
                try:
                    data = sdr_rx.rx()
                    if not isinstance(data, (list, tuple)) or len(data) < 2:
                        raise RuntimeError(
                            "RX returned fewer than two channels; check rx_enabled_channels=[0, 1]"
                        )
                    rx_c0, rx_c1 = data[0], data[1]

                    if DUAL_PLUTO_MODE:
                        raw_cfo = estimate_bpsk_residual_cfo_hz(rx_c0, SAMPLE_RATE)
                        a = DUAL_PLUTO_CFO_EMA_ALPHA
                        cfo_filt_hz = a * raw_cfo + (1.0 - a) * cfo_filt_hz
                        cfo_filt_hz = float(np.clip(cfo_filt_hz, -120e3, 120e3))
                        rx_c0, rx_c1 = derotate_common_cfo(
                            rx_c0, rx_c1, cfo_filt_hz, SAMPLE_RATE
                        )

                    correlation = np.correlate(rx_c0, barker_oversampled, mode="valid")
                    corr_mag = np.abs(correlation)
                    peak_idx = np.argmax(corr_mag)
                    peak_val = corr_mag[peak_idx]
                    avg_energy = np.mean(corr_mag)
                    peak_ratio = peak_val / max(avg_energy, 1e-12)

                    if peak_ratio <= SYNC_PEAK_RATIO_MIN:
                        self._no_sync_loops += 1
                        if self._stream_announced:
                            self._stream_announced = False
                        if self._no_sync_loops == 80:
                            self._have_prev_raw_sample = False
                            self._have_prev_filtered_sample = False
                            self._phase_ema_vec = None
                            self._outlier_count = 0
                        if self._no_sync_loops % 400 == 0:
                            mode = "dual-Pluto+CFO" if DUAL_PLUTO_MODE else "single device"
                            self.signals.status.emit(
                                "RX ok — no Barker lock (check same CENTER_FREQ on TX/RX, "
                                "antennas, TX level; "
                                f"{mode} peak/mean={peak_ratio:.1f} need >{SYNC_PEAK_RATIO_MIN}; "
                                f"CFO est~{cfo_filt_hz:.0f} Hz)"
                            )
                        continue
                    self._no_sync_loops = 0

                    start_idx = peak_idx
                    end_idx = start_idx + packet_len_samples + 200
                    if end_idx >= len(rx_c0):
                        continue

                    pkt_c0 = rx_c0[start_idx:end_idx]
                    pkt_c1 = rx_c1[start_idx:end_idx]

                    sig_power = float(np.mean(np.abs(pkt_c0) ** 2))
                    noise_region = rx_c0[end_idx : end_idx + 500]
                    noise_power = (
                        float(np.mean(np.abs(noise_region) ** 2))
                        if len(noise_region) > 50
                        else sig_power * 1e-3
                    )
                    noise_power = max(noise_power, 1e-12)
                    snr_db = 10.0 * np.log10(max(sig_power / noise_power, 1e-6))

                    seg = min(len(pkt_c0), packet_len_samples)
                    x0 = pkt_c0[:seg]
                    x1 = pkt_c1[:seg]
                    p0 = float(np.sum(np.abs(x0) ** 2))
                    p1 = float(np.sum(np.abs(x1) ** 2))
                    if p0 < 1e-12 or p1 < 1e-12:
                        continue
                    cross_sum = np.sum(x1 * np.conj(x0))
                    coherence = float(np.abs(cross_sum) / np.sqrt(p0 * p1))
                    if coherence < MIN_PHASE_COHERENCE:
                        self._low_coherence_loops += 1
                        if self._low_coherence_loops % 100 == 0:
                            p1_over_p0 = p1 / max(p0, 1e-12)
                            power_hint = (
                                "RX1 weak/dead"
                                if p1_over_p0 < 0.05
                                else "phase unstable/multipath"
                            )
                            self.signals.status.emit(
                                "Barker locked, but RX channel phase coherence is low "
                                f"({coherence:.2f} < {MIN_PHASE_COHERENCE:.2f}); "
                                f"RX1/RX0 power={p1_over_p0:.3f} ({power_hint})."
                            )
                        continue
                    self._low_coherence_loops = 0

                    raw_delta_phi = float(np.angle(cross_sum))
                    with self._state_lock:
                        self._phase_cal_buf.append(raw_delta_phi)
                        phase_cal_offset = self._phase_cal_offset
                        d_spacing = self._d_spacing

                    delta_phi = (raw_delta_phi - phase_cal_offset + np.pi) % (2 * np.pi) - np.pi
                    delta_phi *= PHASE_SIGN
                    phase_vec = np.exp(1j * delta_phi)
                    if self._phase_ema_vec is None:
                        phase_ema_candidate = phase_vec
                    else:
                        a_phase = _clamp(PHASE_EMA_ALPHA, 0.01, 1.0)
                        phase_ema_candidate = (
                            (1.0 - a_phase) * self._phase_ema_vec + a_phase * phase_vec
                        )
                        mag = abs(phase_ema_candidate)
                        if mag < 1e-6:
                            phase_ema_candidate = phase_vec
                        else:
                            phase_ema_candidate /= mag
                    delta_phi = float(np.angle(phase_ema_candidate))

                    # Ch.1 — physical baseline uses self._d_spacing (may load from JSON)
                    sin_theta = delta_phi * WAVELENGTH / (2.0 * np.pi * d_spacing)
                    sin_theta = np.clip(sin_theta, -1.0, 1.0)
                    raw_deg = _clamp(
                        float(np.degrees(np.arcsin(sin_theta))),
                        ANGLE_MIN_DEG,
                        ANGLE_MAX_DEG,
                    )

                    tnow = time.perf_counter()
                    dt = _clamp(tnow - self._last_t, 0.002, 0.5)

                    if self._have_prev_raw_sample:
                        if abs(raw_deg - self._prev_raw_deg) > OUTLIER_THRESHOLD_DEG:
                            self._outlier_count += 1
                            if self._outlier_count < OUTLIER_REACQUIRE_COUNT:
                                continue
                        self._outlier_count = 0
                        max_step = max(0.25, MAX_ANGLE_RATE_DEGPS * dt)
                        raw_delta_deg = raw_deg - self._prev_raw_deg
                        if abs(raw_delta_deg) > max_step:
                            raw_deg = float(self._prev_raw_deg + np.sign(raw_delta_deg) * max_step)
                    self._phase_ema_vec = phase_ema_candidate
                    self._prev_raw_deg = raw_deg
                    self._have_prev_raw_sample = True

                    self._last_t = tnow

                    filt = self.active_filter
                    if isinstance(filt, IMMKalman):
                        filt.set_dt(dt)

                    filtered_deg = filt.step(raw_deg)
                    if self._have_prev_filtered_sample:
                        rate = (filtered_deg - self._prev_filtered) / dt
                    else:
                        rate = 0.0
                        self._have_prev_filtered_sample = True
                    self._prev_filtered = filtered_deg
                    self.signals.new_data.emit(
                        raw_deg,
                        filtered_deg,
                        filt.variance,
                        rate,
                        snr_db,
                        peak_ratio,
                        cfo_filt_hz,
                    )
                    self._error_count = 0
                    if not self._stream_announced:
                        self._stream_announced = True
                        self.signals.status.emit("Barker locked — DoA metrics updating")

                    pkt_corr, _ = correct_frequency_phase(pkt_c0, SAMPLE_RATE)
                    corr_fine = np.correlate(pkt_corr, barker_oversampled, mode="valid")
                    peak_fine = np.argmax(np.abs(corr_fine))
                    data_start = peak_fine + len(barker_oversampled)
                    data_seg = pkt_corr[
                        data_start: data_start + len(data_bits) * SAMPLES_PER_SYMBOL
                    ]
                    symbols = data_seg[SAMPLES_PER_SYMBOL // 2:: SAMPLES_PER_SYMBOL]
                    symbols = symbols * np.exp(-1j * np.angle(corr_fine[peak_fine]))
                    rx_bits = (np.real(symbols) > 0).astype(int)
                    msg = bits_to_text(rx_bits)
                    clean = "".join(c for c in msg if 32 <= ord(c) <= 126)
                    if len(clean) > 3:
                        self.signals.message_decoded.emit(clean, raw_deg, filtered_deg)

                except Exception as e:
                    self._error_count += 1
                    if self._error_count % 20 == 1:
                        self.signals.status.emit(
                            f"SDR error (×{self._error_count}): {e}"
                        )
                        log.warning("SDR loop error: %s", e, exc_info=True)

        finally:
            if sdr_tx is not None:
                try:
                    sdr_tx.tx_enabled_channels = []
                except Exception as e:
                    log.warning("TX stop: %s", e)
                try:
                    destroy = getattr(sdr_tx, "tx_destroy_buffer", None)
                    if destroy is not None:
                        destroy()
                except Exception as e:
                    log.warning("TX buffer destroy: %s", e)
            if sdr_rx is not None:
                try:
                    sdr_rx.rx_enabled_channels = []
                except Exception as e:
                    log.warning("RX stop: %s", e)
            log.info("SDR hardware session released (TX/RX disabled)")


# ═══════════════════════════════════════════════════════════════════════
#  PARAMETER PANELS (one per filter type)
# ═══════════════════════════════════════════════════════════════════════

class GHParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self.spin_g = QDoubleSpinBox(minimum=0.01, maximum=1.0, singleStep=0.05, value=0.40, decimals=2)
        self.spin_h = QDoubleSpinBox(minimum=0.01, maximum=1.0, singleStep=0.05, value=0.10, decimals=2)
        layout.addRow("g (position gain):", self.spin_g)
        layout.addRow("h (velocity gain):", self.spin_h)

    def get_params(self) -> dict:
        return {"g": self.spin_g.value(), "h": self.spin_h.value()}


class Kalman1DParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self.spin_qa = QDoubleSpinBox(minimum=0.01, maximum=50.0, singleStep=0.1, value=0.5, decimals=2)
        self.spin_qv = QDoubleSpinBox(minimum=0.01, maximum=50.0, singleStep=0.1, value=1.0, decimals=2)
        self.spin_r = QDoubleSpinBox(minimum=0.1, maximum=200.0, singleStep=1.0, value=25.0, decimals=1)
        layout.addRow("Q angle σ (°):", self.spin_qa)
        layout.addRow("Q velocity σ (°/s):", self.spin_qv)
        layout.addRow("R meas var (°²):", self.spin_r)

    def get_params(self) -> dict:
        return {
            "q_angle": self.spin_qa.value(),
            "q_vel": self.spin_qv.value(),
            "r_meas": self.spin_r.value(),
        }


class ParticleParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self.spin_n = QSpinBox(minimum=50, maximum=10000, singleStep=100, value=500)
        self.spin_pstd = QDoubleSpinBox(minimum=0.1, maximum=30.0, singleStep=0.5, value=2.0, decimals=1)
        self.spin_mstd = QDoubleSpinBox(minimum=0.1, maximum=30.0, singleStep=0.5, value=5.0, decimals=1)
        layout.addRow("Particle count (N):", self.spin_n)
        layout.addRow("Process noise σ (°):", self.spin_pstd)
        layout.addRow("Measurement noise σ (°):", self.spin_mstd)

    def get_params(self) -> dict:
        return {
            "n_particles": self.spin_n.value(),
            "process_std": self.spin_pstd.value(),
            "meas_std": self.spin_mstd.value(),
        }


class IMMParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        self.spin_r = QDoubleSpinBox(minimum=0.1, maximum=200.0, singleStep=1.0, value=64.0, decimals=1)
        layout.addRow("R meas var (°²):", self.spin_r)

    def get_params(self) -> dict:
        return {"R_deg2": self.spin_r.value()}


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """PyQt5 main window for real-time DoA + multi-filter tracking."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BPSK DoA & Multi-Filter Analyzer")
        self.resize(1100, 700)

        # Empty start — all-NaN buffers trigger pyqtgraph ScatterPlotItem warnings
        self._raw_buf = deque(maxlen=BUFFER_LEN)
        self._filt_buf = deque(maxlen=BUFFER_LEN)
        self._var_buf = deque(maxlen=BUFFER_LEN)

        self._build_ui()
        self._connect_signals()

        self.worker = SDRWorker()
        self.worker.signals.new_data.connect(self._on_new_data)
        self.worker.signals.message_decoded.connect(self._on_message)
        self.worker.signals.status.connect(self._on_status)
        self.worker.start()

        self._plot_timer = QTimer(self)
        self._plot_timer.timeout.connect(self._refresh_plot)
        self._plot_timer.start(80)

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # LEFT — plot area
        plot_frame = QVBoxLayout()
        pg.setConfigOptions(antialias=True, background="#1e1e2e", foreground="#cdd6f4")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "Angle", units="°")
        self.plot_widget.setLabel("bottom", "Sample")
        self.plot_widget.setYRange(ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        self.plot_widget.setXRange(0, BUFFER_LEN)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.addLegend(offset=(10, 10))

        self.curve_raw = self.plot_widget.plot(
            pen=None, symbol="o", symbolSize=3,
            symbolBrush=(255, 80, 80, 120), name="Raw DoA",
        )
        self.curve_filt = self.plot_widget.plot(
            pen=pg.mkPen(color=(80, 250, 123), width=2), name="Filtered",
        )
        self.curve_upper = self.plot_widget.plot(
            pen=pg.mkPen(color=(250, 250, 80, 60), width=1, style=Qt.DashLine),
        )
        self.curve_lower = self.plot_widget.plot(
            pen=pg.mkPen(color=(250, 250, 80, 60), width=1, style=Qt.DashLine),
        )
        self.fill_band = pg.FillBetweenItem(
            self.curve_upper, self.curve_lower,
            brush=pg.mkBrush(250, 250, 80, 25),
        )
        self.plot_widget.addItem(self.fill_band)

        plot_frame.addWidget(self.plot_widget, stretch=1)
        root.addLayout(plot_frame, stretch=3)

        # RIGHT — control panel
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        # Filter selector
        grp_filter = QGroupBox("Filter Selection")
        fl = QVBoxLayout(grp_filter)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(FILTER_REGISTRY.keys())
        self.combo_filter.setCurrentText("IMM Kalman")
        fl.addWidget(self.combo_filter)

        self.param_stack = QStackedWidget()
        self.param_panels = {}
        panel_classes = [GHParamPanel, Kalman1DParamPanel, ParticleParamPanel, IMMParamPanel]
        for (name, _), PanelCls in zip(FILTER_REGISTRY.items(), panel_classes):
            panel = PanelCls()
            self.param_panels[name] = panel
            self.param_stack.addWidget(panel)
        fl.addWidget(self.param_stack)
        self.param_stack.setCurrentIndex(3)  # IMM Kalman default

        self.btn_apply = QPushButton("Apply Parameters")
        self.btn_apply.setStyleSheet(
            "QPushButton{background:#89b4fa;color:#1e1e2e;font-weight:bold;padding:6px;border-radius:4px}"
            "QPushButton:hover{background:#74c7ec}"
        )
        fl.addWidget(self.btn_apply)
        ctrl.addWidget(grp_filter)

        # Metrics panel
        grp_metrics = QGroupBox("Live Metrics")
        ml = QFormLayout(grp_metrics)
        mono = QFont("Monospace", 10)
        self.lbl_raw = QLabel("—")
        self.lbl_raw.setFont(mono)
        self.lbl_filt = QLabel("—")
        self.lbl_filt.setFont(mono)
        self.lbl_var = QLabel("—")
        self.lbl_var.setFont(mono)
        self.lbl_ci = QLabel("—")
        self.lbl_ci.setFont(mono)
        self.lbl_rate = QLabel("—")
        self.lbl_rate.setFont(mono)
        self.lbl_snr = QLabel("—")
        self.lbl_snr.setFont(mono)
        self.lbl_peak = QLabel("—")
        self.lbl_peak.setFont(mono)
        self.lbl_cfo = QLabel("—")
        self.lbl_cfo.setFont(mono)
        self.lbl_active = QLabel("IMM Kalman")
        self.lbl_active.setFont(mono)
        ml.addRow("Active filter:", self.lbl_active)
        ml.addRow("Raw angle (°):", self.lbl_raw)
        ml.addRow("Filtered (°):", self.lbl_filt)
        ml.addRow("Ang. rate (°/s):", self.lbl_rate)
        ml.addRow("SNR (dB):", self.lbl_snr)
        ml.addRow("Barker peak/mean:", self.lbl_peak)
        ml.addRow("CFO track (Hz):", self.lbl_cfo)
        ml.addRow("Variance (°²):", self.lbl_var)
        ml.addRow("95% CI (±°):", self.lbl_ci)
        ctrl.addWidget(grp_metrics)

        # Ch.5 — optional CSV (written from GUI thread in _on_new_data)
        self._csv_file = None
        self._csv_writer = None
        rec_row = QHBoxLayout()
        self.btn_rec = QPushButton("Record CSV")
        self.btn_rec.setCheckable(True)
        self.btn_rec.setStyleSheet(
            "QPushButton{background:#a6e3a1;color:#1e1e2e;padding:4px;border-radius:4px}"
        )
        self.lbl_rec = QLabel("Off")
        self.lbl_rec.setFont(mono)
        rec_row.addWidget(self.btn_rec)
        rec_row.addWidget(self.lbl_rec)
        ctrl.addLayout(rec_row)

        # Calibrate button
        self.btn_cal = QPushButton("Calibrate (Zero)")
        self.btn_cal.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;font-weight:bold;padding:6px;border-radius:4px}"
            "QPushButton:hover{background:#eba0ac}"
        )
        ctrl.addWidget(self.btn_cal)

        # Status bar
        self.lbl_status = QLabel("Waiting for SDR…")
        self.lbl_status.setFrameStyle(QFrame.StyledPanel)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.lbl_status)

        # Decoded message label
        grp_msg = QGroupBox("Last Decoded Message")
        msg_l = QVBoxLayout(grp_msg)
        self.lbl_msg = QLabel("—")
        self.lbl_msg.setFont(QFont("Monospace", 11, QFont.Bold))
        self.lbl_msg.setAlignment(Qt.AlignCenter)
        msg_l.addWidget(self.lbl_msg)
        ctrl.addWidget(grp_msg)

        ctrl.addStretch()
        root.addLayout(ctrl, stretch=1)

    # ── signal wiring ─────────────────────────────────────────────────

    def _connect_signals(self):
        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cal.clicked.connect(self._on_calibrate)
        self.btn_rec.toggled.connect(self._on_recording_toggled)

    def _on_filter_changed(self, idx):
        self.param_stack.setCurrentIndex(idx)

    def _on_apply(self):
        name = self.combo_filter.currentText()
        panel = self.param_panels[name]
        params = panel.get_params()
        cls = FILTER_REGISTRY[name][0]
        new_filter = cls(**params)
        self.worker.active_filter = new_filter
        self.lbl_active.setText(name)
        self.lbl_status.setText(f"Switched to {name} with new params")

    def _on_calibrate(self):
        self.worker.calibrate()

    def _on_recording_toggled(self, on: bool):
        if on:
            try:
                CSV_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
                path = CSV_EXPORT_DIR / f"doa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self._csv_file = open(path, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(
                    [
                        "unix_t",
                        "raw_deg",
                        "filtered_deg",
                        "variance",
                        "rate_degps",
                        "snr_db",
                        "barker_peak_over_mean",
                        "cfo_hz",
                    ]
                )
                self.lbl_rec.setText(path.name)
                self.lbl_status.setText(f"Recording → {path}")
            except OSError as e:
                self.btn_rec.setChecked(False)
                self.lbl_status.setText(f"CSV open failed: {e}")
        else:
            if self._csv_file is not None:
                try:
                    self._csv_file.close()
                except OSError:
                    pass
            self._csv_file = None
            self._csv_writer = None
            self.lbl_rec.setText("Off")

    # ── data slots (GUI thread) ───────────────────────────────────────

    def _on_new_data(
        self,
        raw: float,
        filtered: float,
        var: float,
        rate: float,
        snr_db: float,
        peak_ratio: float,
        cfo_hz: float,
    ):
        self._raw_buf.append(raw)
        self._filt_buf.append(filtered)
        self._var_buf.append(var)

        self.lbl_raw.setText(f"{raw:+7.2f}")
        self.lbl_filt.setText(f"{filtered:+7.2f}")
        self.lbl_rate.setText(f"{rate:+7.1f}")
        self.lbl_snr.setText(f"{snr_db:.1f}")
        self.lbl_peak.setText(f"{peak_ratio:5.2f}")
        self.lbl_cfo.setText(f"{cfo_hz:8.0f}")
        self.lbl_var.setText(f"{var:.2f}")
        ci = 1.96 * np.sqrt(max(var, 0))
        self.lbl_ci.setText(f"±{ci:.2f}")
        w = self._csv_writer
        if w is not None:
            w.writerow(
                [time.time(), raw, filtered, var, rate, snr_db, peak_ratio, cfo_hz]
            )

    def _on_message(self, msg: str, raw: float, filt: float):
        self.lbl_msg.setText(msg)

    def _on_status(self, text: str):
        self.lbl_status.setText(text)

    # ── plot refresh (timer-driven, never blocks SDR) ─────────────────

    def _refresh_plot(self):
        n = len(self._raw_buf)
        if n == 0:
            self.curve_raw.setData([], [])
            self.curve_filt.setData([], [])
            self.curve_upper.setData([], [])
            self.curve_lower.setData([], [])
            return

        raw = np.asarray(self._raw_buf, dtype=np.float64)
        filt = np.asarray(self._filt_buf, dtype=np.float64)
        var = np.asarray(self._var_buf, dtype=np.float64)
        xs = np.arange(n, dtype=np.float64)

        self.curve_raw.setData(xs, raw)
        self.curve_filt.setData(xs, filt)

        ci = 1.96 * np.sqrt(np.maximum(var, 0.0))
        self.curve_upper.setData(xs, filt + ci)
        self.curve_lower.setData(xs, filt - ci)

    # ── cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.btn_rec.isChecked():
            self.btn_rec.setChecked(False)
        self.worker.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Ch.2 — stderr logging (ops / thesis appendix)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    from PyQt5.QtGui import QColor
    palette.setColor(palette.Window, QColor("#1e1e2e"))
    palette.setColor(palette.WindowText, QColor("#cdd6f4"))
    palette.setColor(palette.Base, QColor("#313244"))
    palette.setColor(palette.AlternateBase, QColor("#45475a"))
    palette.setColor(palette.Text, QColor("#cdd6f4"))
    palette.setColor(palette.Button, QColor("#45475a"))
    palette.setColor(palette.ButtonText, QColor("#cdd6f4"))
    palette.setColor(palette.Highlight, QColor("#89b4fa"))
    palette.setColor(palette.HighlightedText, QColor("#1e1e2e"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
