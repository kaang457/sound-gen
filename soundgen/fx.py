"""Post-generation effects rack (like a DAW plugin): applied to the generated audio, exact and fast.

    chain: duration (time-stretch -> trim -> fades) -> intensity (gain) -> distance (level + air absorption)
           -> environment (convolution reverb, algorithmic stereo impulse responses) -> stereo position (pan)
           -> safety limiter
    process(x, sr, Rack(...)) -> (stereo [2, T], mono [T])

Design notes
  * Intensity is a plain gain in dB (the mel controls in controls.py are the generative counterpart).
  * Distance d in 0..1 maps to 1..40 m: inverse-distance level (-6 dB per doubling, capped), a 2nd-order low-pass
    for air absorption (20 kHz at 1 m down to ~3 kHz at 40 m), and more reverb relative to the direct sound.
  * Reverb: impulse responses are synthesised, not recorded: sparse early reflections (first ~80 ms) + a dense
    noise tail with frequency-dependent decay (highs die faster, three bands with their own RT60), independent
    noise per channel (natural stereo width), energy-normalised so "wet" means the same loudness in every room.
    Convolution via FFT. No feedback loops, so no metallic ringing.
  * Time-stretch keeps pitch: phase vocoder with identity phase locking (Laroche & Dolson), transients slightly
    softened at large ratios.
  * Stereo output pans the dry/direct path with an equal-power law; the reverb stays wide. Mono output is the sum.
"""
from dataclasses import dataclass, asdict

import numpy as np

ENVIRONMENTS = {
    # rt60 (s) low / mid / high band, pre-delay (s), early reflections (count, spread s), default wet
    "dry":     None,
    "room":    dict(rt=(0.55, 0.45, 0.30), pre=0.004, er=(6, 0.025), wet=0.22),
    "hall":    dict(rt=(2.2, 1.8, 1.1), pre=0.020, er=(10, 0.070), wet=0.30),
    "cave":    dict(rt=(3.6, 3.0, 1.6), pre=0.035, er=(14, 0.110), wet=0.38),
    "outdoor": dict(rt=(0.35, 0.25, 0.15), pre=0.060, er=(3, 0.180), wet=0.12),   # sparse, a far wall / ground
}


@dataclass
class Rack:
    intensity_db: float = 0.0       # -24 .. +12
    distance: float = 0.0           # 0 (1 m) .. 1 (40 m)
    environment: str = "dry"        # dry | room | hall | cave | outdoor
    wet: float = -1.0               # reverb amount 0..1; -1 = the environment's default
    pan: float = 0.0                # -1 left .. +1 right
    stretch: float = 1.0            # duration factor 0.5 .. 2 (pitch kept)
    trim_start: float = 0.0         # seconds cut from the start
    trim_end: float = 0.0           # seconds cut from the end
    fade_in_ms: float = 5.0
    fade_out_ms: float = 30.0
    tail_s: float = -1.0            # extra time for the reverb tail; -1 = automatic

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ building blocks
def gain(x, db):
    return x * (10.0 ** (db / 20.0))


def _sos_lowpass(sr, fc, order=2):
    from scipy.signal import butter
    return butter(order, min(fc, 0.45 * sr), "lowpass", fs=sr, output="sos")


def distance_params(d):
    """d 0..1 -> (metres, level dB, air low-pass cutoff Hz, reverb send multiplier)."""
    d = float(np.clip(d, 0.0, 1.0))
    m = 1.0 * 40.0 ** d                                   # 1 m .. 40 m, perceptually even
    level = -20.0 * np.log10(m) * 0.8                     # a little gentler than free field (-6 dB / doubling)
    level = max(level, -26.0)
    fc = 20000.0 / (1.0 + 0.15 * (m - 1.0))               # ~20 kHz at 1 m, ~3 kHz at 40 m
    send = 1.0 + 1.5 * d                                  # far sources are relatively more reverberant
    return m, level, fc, send


def distance(x, sr, d):
    from scipy.signal import sosfilt
    if d <= 0:
        return x
    _, level, fc, _ = distance_params(d)
    y = sosfilt(_sos_lowpass(sr, fc), x, axis=-1)
    return gain(y, level)


def impulse_response(sr, env, seed=0):
    """Synthetic stereo IR [2, N] for an environment preset (see module notes)."""
    from scipy.signal import butter, sosfilt
    p = ENVIRONMENTS[env]
    rng = np.random.default_rng(seed)
    rt = p["rt"]
    n = int((max(rt) * 1.1 + p["pre"] + p["er"][1]) * sr)
    t = np.arange(n) / sr
    ir = np.zeros((2, n))
    bands = [butter(2, 300, "lowpass", fs=sr, output="sos"),
             butter(2, [300, 3000], "bandpass", fs=sr, output="sos"),
             butter(2, 3000, "highpass", fs=sr, output="sos")]
    start = int((p["pre"] + 0.5 * p["er"][1]) * sr)
    for ch in range(2):
        noise = rng.standard_normal(n)
        tail = np.zeros(n)
        for sos, r in zip(bands, rt):
            env_ = np.exp(-6.9078 * t / r)                # -60 dB at t = rt60
            tail += sosfilt(sos, noise) * env_
        ramp = np.clip((np.arange(n) - start) / max(1, int(0.03 * sr)), 0, 1)   # tail fades in after the ERs
        ir[ch] = tail * ramp
        k, spread = p["er"]
        times = p["pre"] + np.sort(rng.uniform(0, spread, k))
        amps = rng.uniform(0.35, 0.8, k) * np.exp(-3.0 * (times - p["pre"]) / max(spread, 1e-3))
        for tm, a in zip(times, amps):
            i = int(tm * sr)
            if i < n:
                ir[ch, i] += a * rng.choice([-1, 1]) * 4.0
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)) + 1e-12   # unit energy per channel
    return ir


_IR_CACHE = {}


def reverb(x, sr, env, wet):
    """mono x [T] -> stereo wet signal [2, T + len(IR) - 1] (only the reverberant part)."""
    from scipy.signal import fftconvolve
    if env == "dry" or ENVIRONMENTS.get(env) is None or wet <= 0:
        return None
    key = (sr, env)
    if key not in _IR_CACHE:
        _IR_CACHE[key] = impulse_response(sr, env)
    ir = _IR_CACHE[key]
    return np.stack([fftconvolve(x, ir[c]) for c in range(2)]) * wet


def pan(x, p):
    """mono [T] -> stereo [2, T], equal-power law."""
    a = (float(np.clip(p, -1, 1)) + 1.0) * np.pi / 4.0
    return np.stack([x * np.cos(a), x * np.sin(a)])


def time_stretch(x, factor, n_fft=2048, hop=512):
    """Duration x factor, pitch unchanged (phase vocoder with identity phase locking)."""
    import torch
    if abs(factor - 1.0) < 1e-3 or len(x) < n_fft:
        return x
    win = torch.hann_window(n_fft, dtype=torch.float64)
    X = torch.stft(torch.from_numpy(np.asarray(x, np.float64)), n_fft, hop, window=win, return_complex=True).numpy()
    F_, T_ = X.shape
    steps = np.arange(0, T_ - 1, 1.0 / factor)
    omega = 2 * np.pi * hop * np.arange(F_) / n_fft
    phase = np.angle(X[:, 0])
    out = np.zeros((F_, len(steps)), np.complex128)
    for k, s in enumerate(steps):
        i = int(s)
        frac = s - i
        a, b = X[:, i], X[:, i + 1]
        mag = (1 - frac) * np.abs(a) + frac * np.abs(b)
        dphi = np.angle(b) - np.angle(a) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        # identity phase locking: bins follow the phase of their spectral peak (keeps transients/tones coherent)
        peaks = np.r_[False, (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]), False]
        if peaks.any():
            pk = np.flatnonzero(peaks)
            region = pk[np.clip(np.searchsorted((pk[:-1] + pk[1:]) / 2, np.arange(F_)), 0, len(pk) - 1)]
            locked = phase[region] + (np.angle(a) - np.angle(a)[region])
        else:
            locked = phase
        out[:, k] = mag * np.exp(1j * locked)
        phase = phase + omega + dphi
    y = torch.istft(torch.from_numpy(out), n_fft, hop, window=win,
                    length=int(round(len(x) * factor))).numpy()
    return y


def fades(x, sr, fade_in_ms, fade_out_ms):
    x = x.copy()
    n_in, n_out = int(sr * fade_in_ms / 1000), int(sr * fade_out_ms / 1000)
    if n_in > 0:
        x[..., :n_in] *= np.linspace(0, 1, min(n_in, x.shape[-1]))[: x.shape[-1]] ** 2
    if n_out > 0:
        k = min(n_out, x.shape[-1])
        x[..., -k:] *= np.linspace(1, 0, k) ** 2
    return x


def limiter(x, ceiling=0.97):
    """Soft safety limiter: transparent below ~0.8, never clips."""
    peak = np.abs(x).max()
    if peak <= 0.8:
        return x
    knee = 0.8
    y = np.where(np.abs(x) <= knee, x,
                 np.sign(x) * (knee + (ceiling - knee) * np.tanh((np.abs(x) - knee) / (ceiling - knee))))
    return y


# ------------------------------------------------------------------ the rack
def process(x, sr, rack: Rack):
    """mono x [T] -> (stereo [2, T'], mono [T'])."""
    r = rack
    x = np.asarray(x, np.float64)
    x = time_stretch(x, float(np.clip(r.stretch, 0.25, 4.0)))
    a, b = int(max(0, r.trim_start) * sr), int(max(0, r.trim_end) * sr)
    x = x[a: len(x) - b if b > 0 else None]
    if len(x) == 0:
        return np.zeros((2, 1)), np.zeros(1)
    x = fades(x, sr, r.fade_in_ms, r.fade_out_ms)
    x = gain(x, r.intensity_db)
    direct = distance(x, sr, r.distance)
    env = r.environment if r.environment in ENVIRONMENTS else "dry"
    p = ENVIRONMENTS.get(env)
    wet_amt = (p["wet"] if (p and r.wet < 0) else max(r.wet, 0.0)) * (distance_params(r.distance)[3] if p else 0)
    # reverb is fed by the distance-filtered direct sound, then scaled so that wet ~ direct energy * wet_amt
    wet = reverb(direct, sr, env, 1.0) if p else None
    st = pan(direct, r.pan)
    if wet is not None:
        tail = int((max(p["rt"]) * 0.8) * sr) if r.tail_s < 0 else int(r.tail_s * sr)
        n = min(wet.shape[1], st.shape[1] + tail)
        e_d = np.sqrt((direct ** 2).mean()) + 1e-12
        e_w = np.sqrt((wet ** 2).mean()) + 1e-12
        wet = wet[:, :n] * (e_d / e_w) * wet_amt
        st = np.pad(st, ((0, 0), (0, n - st.shape[1])))
        st = st + wet
        k = min(int(0.25 * sr), n // 4)                   # gentle end of the tail
        st[:, n - k:] *= np.linspace(1, 0, k) ** 2
    mono = np.pad(direct, (0, st.shape[1] - len(direct)))
    if wet is not None:
        mono = mono + wet.mean(0)
    return limiter(st).astype(np.float32), limiter(mono).astype(np.float32)
