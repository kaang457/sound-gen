"""Audio I/O and the low-resolution mel "sketch" used to condition WaveFlow.

The sketch follows Andreu & Villanueva (AIIDE 2022): a coarse log-mel
spectrogram (10-30 bands). It keeps the loudness envelope and rough spectral
shape, and leaves the fine detail to the generative model. Because it is
low-dimensional and interpretable, it is also the surface our parametric
controls edit (see controls.py).

Normalisation is ABSOLUTE (not per clip): sketch = (dB + 100) / 100.
Loudness information therefore survives, which the intensity and distance
controls depend on.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

SR = 22050
DB_FLOOR = -100.0


@dataclass
class MelConfig:
    sample_rate: int = SR
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 20
    f_min: float = 30.0
    f_max: float = 11025.0

    def to_dict(self):
        return asdict(self)


class MelSketch(nn.Module):
    """waveform [B, T] in [-1, 1] -> sketch [B, n_mels, T // hop + 1] in ~[0, 1.2]."""

    def __init__(self, cfg: MelConfig = MelConfig()):
        super().__init__()
        import torchaudio
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.n_fft,
            hop_length=cfg.hop, f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels,
            power=2.0, center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney")

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            power = self.mel(wav.float())
            db = 10.0 * torch.log10(power.clamp_min(1e-10))
            return db_to_sketch(db)


def db_to_sketch(db):
    if isinstance(db, torch.Tensor):
        return ((db - DB_FLOOR) / 100.0).clamp(0.0, 1.2)
    return np.clip((db - DB_FLOOR) / 100.0, 0.0, 1.2)


def sketch_to_db(sketch):
    return sketch * 100.0 + DB_FLOOR


# ---------------------------------------------------------------- file I/O
def load_audio(path, sr: int = SR) -> np.ndarray:
    """Any format soundfile/librosa can read -> mono float32 at `sr`."""
    try:
        import soundfile as sf
        x, rate = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        import librosa  # handles mp3 when libsndfile cannot
        x, rate = librosa.load(str(path), sr=None, mono=True)
        x = x.astype(np.float32)
    if rate != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(rate), int(sr))
        x = resample_poly(x, sr // g, int(rate) // g).astype(np.float32)
    return x


def save_audio(path, x: np.ndarray, sr: int = SR):
    import soundfile as sf
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12))


def trim_silence(x: np.ndarray, sr: int = SR, threshold_db: float = -50.0, pad_s: float = 0.05):
    """Trim leading/trailing parts quieter than threshold (relative to peak)."""
    if len(x) == 0:
        return x
    frame = max(1, int(0.01 * sr))
    n = len(x) // frame
    if n == 0:
        return x
    env = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    ref = env.max()
    active = np.where(20 * np.log10(env / ref + 1e-12) > threshold_db)[0]
    if len(active) == 0:
        return x[:0]
    pad = int(pad_s * sr)
    start = max(0, active[0] * frame - pad)
    end = min(len(x), (active[-1] + 1) * frame + pad)
    return x[start:end]
