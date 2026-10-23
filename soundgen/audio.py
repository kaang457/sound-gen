"""Audio I/O and the low-resolution mel "sketch" used to condition WaveFlow.

The sketch follows Andreu & Villanueva (AIIDE 2022): a coarse log-mel
spectrogram (10-30 bands). It keeps the loudness envelope and rough spectral
shape, and leaves the fine detail to the generative model. Because it is
low-dimensional and interpretable, it is also the surface our parametric
controls edit (see controls.py).

MEL CONTRACT (v3, 23 Oct 2026) = NVIDIA BigVGAN-v2 `bigvgan_v2_44khz_128band_512x`, so the pretrained vocoder
reads our mels directly. Every module speaks it: the mel generator produces it, the controls edit it, a vocoder
(BigVGAN-v2, or our WaveFlow trained on the same contract) turns it into audio.

    44.1 kHz mono · n_fft 2048 (hann 2048) · hop 512 (86.1 frames/s; T samples -> T / 512 frames)
    128 mel bands 0 Hz - 22.05 kHz (librosa slaney) · magnitude, reflect pad (n_fft - hop) / 2, no centring
    BigVGAN log-mel  L = ln(max(mel, 1e-5))   <->   our sketch units  s = (20 log10(mel) + 100) / 100
    (so the BigVGAN floor 1e-5 is exactly our -100 dB floor; full-scale tones reach ~ +25 dB = 1.25)

Older contracts stay loadable because every checkpoint stores its own mel_cfg (a dict without "style" is the
old power-dB mel): MEL_V2 (48 kHz, hop 480, 21 Oct, never trained) and MEL_POC (22.05 kHz, 20 bands, poc_v1).
Packs may be at any rate: MelSketch(cfg, in_sr=48000) resamples on the fly.

Normalisation is ABSOLUTE (not per clip): sketch = (dB + 100) / 100.
Loudness information therefore survives, which the intensity and distance
controls depend on.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

SR = 44100                     # contract v3 (BigVGAN-v2 44 kHz); packs from 01 are 48 kHz and get resampled
SKETCH_MAX = 1.5               # +50 dB; tones at full scale reach ~1.25 in contract v3
LN10_20 = 20.0 / np.log(10.0)  # natural-log magnitude -> dB
DB_FLOOR = -100.0


@dataclass
class MelConfig:
    sample_rate: int = SR
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 22050.0
    style: str = "bigvgan"      # "bigvgan": magnitude, ln, no centring (contract v3) | "power_db": older contracts

    def to_dict(self):
        return asdict(self)

    @property
    def fps(self):
        return self.sample_rate / self.hop


MEL_V3 = MelConfig()
MEL_V2 = MelConfig(sample_rate=48000, n_fft=2048, hop=480, n_mels=128, f_min=20.0, f_max=24000.0, style="power_db")
MEL_POC = MelConfig(sample_rate=22050, n_fft=1024, hop=256, n_mels=20, f_min=30.0, f_max=11025.0, style="power_db")


def mel_config(d):
    """A stored mel_cfg dict -> MelConfig. Dicts saved before contract v3 have no "style": they are power-dB mels."""
    if isinstance(d, MelConfig):
        return d
    d = dict(d)
    d.setdefault("style", "power_db")
    return MelConfig(**d)


def _slaney_fbank(cfg):
    try:
        import librosa
        fb = librosa.filters.mel(sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.f_min,
                                 fmax=cfg.f_max)
        return torch.from_numpy(fb).float()
    except ImportError:                        # torchaudio's slaney filters are the same (checked: max diff ~1e-8)
        import torchaudio
        return torchaudio.functional.melscale_fbanks(cfg.n_fft // 2 + 1, cfg.f_min, cfg.f_max, cfg.n_mels,
                                                     cfg.sample_rate, norm="slaney", mel_scale="slaney").T.contiguous()


class MelSketch(nn.Module):
    """waveform [B, T] in [-1, 1] -> sketch [B, n_mels, F] in sketch units (0 = -100 dB).
    contract v3 ("bigvgan"): F = T // hop, identical to BigVGAN's get_mel_spectrogram after the unit change.
    older ("power_db"): F = T // hop + 1. in_sr: rate of the incoming audio if it is not cfg.sample_rate."""

    def __init__(self, cfg: MelConfig = MelConfig(), in_sr=None):
        super().__init__()
        import torchaudio
        self.cfg = cfg = mel_config(cfg)
        self.resample = (torchaudio.transforms.Resample(in_sr, cfg.sample_rate)
                         if in_sr and int(in_sr) != cfg.sample_rate else None)
        if cfg.style == "bigvgan":
            self.register_buffer("fb", _slaney_fbank(cfg))
            self.register_buffer("win", torch.hann_window(cfg.n_fft))
        else:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.n_fft,
                hop_length=cfg.hop, f_min=cfg.f_min, f_max=cfg.f_max, n_mels=cfg.n_mels,
                power=2.0, center=True, pad_mode="reflect", norm="slaney", mel_scale="slaney")

    def log_mel(self, wav):
        """BigVGAN's own log-mel (natural log of the clamped magnitude mel)."""
        c = self.cfg
        p = (c.n_fft - c.hop) // 2
        y = torch.nn.functional.pad(wav[:, None], (p, p), mode="reflect")[:, 0]
        spec = torch.stft(y, c.n_fft, hop_length=c.hop, win_length=c.n_fft, window=self.win, center=False,
                          return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        return torch.log(torch.clamp(self.fb @ spec, min=1e-5))

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = wav.float()
            if self.resample is not None:
                wav = self.resample.to(wav.device)(wav)
            if self.cfg.style == "bigvgan":
                return db_to_sketch(self.log_mel(wav) * LN10_20)
            power = self.mel(wav)
            db = 10.0 * torch.log10(power.clamp_min(1e-10))
            return db_to_sketch(db)


def sketch_to_logmel(sketch):
    """sketch units -> BigVGAN log-mel (what the pretrained vocoder expects)."""
    return (sketch * 100.0 + DB_FLOOR) / LN10_20


def db_to_sketch(db):
    if isinstance(db, torch.Tensor):
        return ((db - DB_FLOOR) / 100.0).clamp(0.0, SKETCH_MAX)
    return np.clip((db - DB_FLOOR) / 100.0, 0.0, SKETCH_MAX)


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
