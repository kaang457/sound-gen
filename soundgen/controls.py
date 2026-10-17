"""Parametric controls = deterministic edits of the low-resolution mel sketch.

The generative model turns any sketch into detailed audio, so editing the sketch
is how parameters reach the sound. Nothing here is learned, so these edits add no
model error to the chain; they are perceptual, not physically calibrated.

    intensity   -1..1   quieter/softer  ..  louder/brighter   (+-12 dB, brightness tilt)
    weight      -1..1   light           ..  heavy             (low-frequency tilt)
    distance     0..1   close           ..  far               (level, air absorption, more reverb)
    environment         dry | room | hall | outdoor | cave    (reverb tail on the sketch)
    duration    seconds or None                                (loop textures, pad/trim events)
    speed       0.5..2  event rate for sequences (footsteps, gallop), time-scale otherwise
"""
from dataclasses import dataclass, asdict

import numpy as np

from .audio import DB_FLOOR

ENVIRONMENTS = {            # RT60 (s), wet level relative to direct (dB), extra HF loss (dB)
    "dry": (0.0, -99.0, 0.0),
    "room": (0.5, -10.0, 0.0),
    "hall": (1.8, -6.0, 0.0),
    "outdoor": (0.25, -16.0, 4.0),
    "cave": (3.0, -3.0, 2.0),
}


@dataclass
class Controls:
    intensity: float = 0.0
    weight: float = 0.0
    distance: float = 0.0
    environment: str = "dry"
    duration: float = None
    speed: float = 1.0

    def to_dict(self):
        return asdict(self)


def band_position(n_mels, f_min=30.0, f_max=11025.0):
    """0..1 position of each mel band on a log-frequency axis (for spectral tilts)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    centers = imel(np.linspace(mel(f_min), mel(f_max), n_mels + 2)[1:-1])
    lf = np.log(centers)
    return (lf - lf[0]) / (lf[-1] - lf[0])


def apply_controls(sketch, controls: Controls, kind="event", mel_cfg=None, seed=0):
    """sketch [n_mels, F] in 0..1.2 -> edited sketch (numpy float32)."""
    c = controls
    mel_cfg = mel_cfg or dict(sample_rate=22050, hop=256, f_min=30.0, f_max=11025.0)
    fps = mel_cfg["sample_rate"] / mel_cfg["hop"]
    rng = np.random.default_rng(seed)
    s = np.asarray(sketch, dtype=np.float32)
    db = s * 100.0 + DB_FLOOR
    pos = band_position(db.shape[0], mel_cfg.get("f_min", 30.0), mel_cfg.get("f_max", 11025.0))[:, None]

    # ---- time: speed, then duration
    if abs(c.speed - 1.0) > 1e-3:
        db = respace_events(db, c.speed, fps) if kind == "sequence" else time_stretch(db, c.speed)
    if c.duration:
        db = fit_duration(db, int(round(c.duration * fps)) + 1, kind, fps, rng)

    # ---- level and spectrum
    gain = 12.0 * c.intensity + 2.0 * c.weight
    tilt = 4.0 * c.intensity * pos + 8.0 * c.weight * (1 - 2 * pos)
    d = float(np.clip(c.distance, 0, 1))
    gain -= 20 * np.log10(1 + 9 * d)
    air = -18.0 * d * np.clip((pos - 0.3) / 0.7, 0, 1)
    db = np.where(db > DB_FLOOR + 1, db + gain + tilt + air, db)

    # ---- environment (+ distance adds diffuse sound)
    rt60, wet_db, hf_loss = ENVIRONMENTS.get(c.environment, ENVIRONMENTS["dry"])
    if d > 0:
        rt60 = max(rt60, 0.3 + 1.0 * d)
        wet_db = max(wet_db, -14.0 + 10.0 * d)
    if rt60 > 0:
        db = add_reverb(db, rt60, wet_db, fps, extend=c.duration is None)
        db = db - hf_loss * pos
    return np.clip((db - DB_FLOOR) / 100.0, 0.0, 1.2).astype(np.float32)


# ------------------------------------------------------------------ primitives
def time_stretch(db, speed):
    F = db.shape[1]
    n = max(2, int(round(F / speed)))
    src = np.linspace(0, F - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, F - 1)
    w = src - i0
    return db[:, i0] * (1 - w) + db[:, i1] * w


def onsets(db, fps, min_gap_s=0.08, rise_db=6.0):
    e = 10 * np.log10(np.mean(10 ** (db / 10), axis=0) + 1e-12)
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    rise = np.diff(e, prepend=e[0])
    peaks, last = [], -10 ** 9
    for t in range(1, len(e) - 1):
        if rise[t] > rise_db and rise[t] >= rise[t - 1] and rise[t] >= rise[t + 1] and t - last >= min_gap_s * fps:
            peaks.append(t)
            last = t
    return peaks


def respace_events(db, speed, fps):
    """Keep each event's shape, change the time between events (footsteps, gallop)."""
    on = onsets(db, fps)
    if len(on) < 2:
        return time_stretch(db, speed)
    on = [max(0, o - 1) for o in on]
    F = db.shape[1]
    first = on[0]
    new_pos = [first + int(round((o - first) / speed)) for o in on]
    seg_end = on[1:] + [F]
    length = new_pos[-1] + (F - on[-1])
    out = np.full((db.shape[0], length), DB_FLOOR, dtype=np.float32)
    out[:, :first] = db[:, :first]
    for o, e, p in zip(on, seg_end, new_pos):
        seg = db[:, o:e]
        # an event may ring beyond the next onset when slowed down: let it decay naturally
        n = min(seg.shape[1], length - p)
        out[:, p:p + n] = np.maximum(out[:, p:p + n], seg[:, :n])
    return out


def fit_duration(db, n, kind, fps, rng):
    F = db.shape[1]
    if n == F:
        return db
    if n < F:
        start = int(rng.integers(0, F - n + 1)) if kind == "texture" else 0
        out = db[:, start:start + n].copy()
        fade = min(n // 4, int(0.15 * fps))
        if fade > 1:
            out[:, -fade:] += np.linspace(0, -40, fade)[None]
        return out
    if kind == "event":
        pad = np.full((db.shape[0], n - F), DB_FLOOR, dtype=np.float32)
        return np.concatenate([db, pad], axis=1)
    on = onsets(db, fps) if kind == "sequence" else []
    if len(on) >= 2:
        # sequences: repeat whole event cycles so the rhythm stays regular at the seam
        a, b = max(0, on[0] - 1), on[-1] - 1
        unit, parts, total = db[:, a:b], [db[:, :a]], a
        while total < n:
            parts.append(unit)
            total += unit.shape[1]
        return fit_duration(np.concatenate(parts, axis=1), n, "event", fps, rng)
    # texture / sequence: loop with an equal-power crossfade in the linear domain
    xf = max(2, min(int(0.3 * fps), F // 3))
    out = 10 ** (db / 10)
    loop = out
    while out.shape[1] < n:
        a, b = np.sqrt(np.linspace(1, 0, xf)), np.sqrt(np.linspace(0, 1, xf))
        mixed = out[:, -xf:] * a ** 2 + loop[:, :xf] * b ** 2
        out = np.concatenate([out[:, :-xf], mixed, loop[:, xf:]], axis=1)
    return 10 * np.log10(out[:, :n] + 1e-12)


def add_reverb(db, rt60, wet_db, fps, extend=True):
    """Exponential decay smearing on the power sketch: a reverb tail at sketch level."""
    k_len = max(2, int(rt60 * 1.2 * fps))
    k = 10 ** (-6.0 * np.arange(1, k_len + 1) / (rt60 * fps))       # -60 dB at rt60
    k = k / k.sum()
    p = 10 ** (db / 10)
    if extend:
        p = np.concatenate([p, np.zeros((p.shape[0], k_len), np.float32)], axis=1)
    wet = np.stack([np.convolve(row, np.concatenate([[0.0], k]))[: p.shape[1]] for row in p])
    out = p + wet * 10 ** (wet_db / 10)
    db2 = 10 * np.log10(out + 1e-10)
    return np.maximum(db2, DB_FLOOR)
