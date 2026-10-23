"""Vocoders: mel (sketch units, mel contract v3) -> audio. Interchangeable, same interface:

    voc = load_vocoder("bigvgan")                 # NVIDIA BigVGAN-v2 44 kHz (pretrained, MIT)
    voc = load_vocoder("runs/vocoder_v1/best.pt") # our WaveFlow trained as a universal vocoder on the same contract
    audio = voc(sketch)                           # sketch [n_mels, F] or [B, n_mels, F] -> np [B, T] at voc.sr

BigVGAN's code is not on PyPI: it is cloned from GitHub once (NVIDIA/BigVGAN) and its weights come from the
Hugging Face hub (nvidia/bigvgan_v2_44khz_128band_512x). Our mel matches its get_mel_spectrogram exactly
(see audio.py); only the units differ (sketch_to_logmel).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .audio import MEL_V3, mel_config, sketch_to_logmel

BIGVGAN_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
BIGVGAN_GIT = "https://github.com/NVIDIA/BigVGAN"


def bigvgan_code(dest=None):
    """Clone NVIDIA/BigVGAN (code only, ~1 MB) and make its modules importable."""
    dest = Path(dest or Path.home() / ".cache" / "BigVGAN")
    if not (dest / "bigvgan.py").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", BIGVGAN_GIT, str(dest)], check=True)
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    return dest


class BigVGANVocoder:
    kind = "bigvgan"

    def __init__(self, model_id=BIGVGAN_ID, device=None, model=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if model is None:
            bigvgan_code()
            import bigvgan
            model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False)
        h = model.h
        want = dict(sampling_rate=MEL_V3.sample_rate, n_fft=MEL_V3.n_fft, hop_size=MEL_V3.hop,
                    win_size=MEL_V3.n_fft, num_mels=MEL_V3.n_mels, fmin=MEL_V3.f_min)
        bad = {k: (h.get(k), v) for k, v in want.items() if h.get(k) is not None and float(h.get(k)) != float(v)}
        if bad or h.get("fmax") not in (None, MEL_V3.f_max):
            raise ValueError(f"BigVGAN config does not match mel contract v3: {bad}, fmax {h.get('fmax')}")
        model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        self.mel_cfg = MEL_V3.to_dict()
        self.sr = MEL_V3.sample_rate
        self.step = None

    @torch.no_grad()
    def __call__(self, sketch, **_):
        s = torch.as_tensor(np.asarray(sketch), dtype=torch.float32)
        if s.dim() == 2:
            s = s[None]
        y = self.model(sketch_to_logmel(s).to(self.device))             # [B, 1, F * hop]
        return y[:, 0].clamp(-1, 1).float().cpu().numpy()


class WaveFlowVocoder:
    kind = "waveflow"

    def __init__(self, ckpt, device=None, temperature=1.0):
        from .train import load_model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_model(ckpt, self.device)
        self.mel_cfg = mel_config(ck["mel_cfg"]).to_dict()
        self.sr = self.mel_cfg["sample_rate"]
        self.vocoder = bool((ck.get("train_cfg") or {}).get("vocoder"))
        self.temperature, self.step = temperature, ck.get("step")

    @torch.no_grad()
    def __call__(self, sketch, seed=0, label=0, **_):
        s = torch.as_tensor(np.asarray(sketch), dtype=torch.float32)
        if s.dim() == 2:
            s = s[None]
        g = torch.Generator(device=self.device).manual_seed(seed)
        y = torch.full((s.shape[0],), label, dtype=torch.long, device=self.device)
        return self.model.sample(s.to(self.device), y, temperature=self.temperature, generator=g).float().cpu().numpy()


def load_vocoder(spec="bigvgan", device=None):
    if isinstance(spec, (BigVGANVocoder, WaveFlowVocoder)):
        return spec
    if str(spec).lower() == "bigvgan":
        return BigVGANVocoder(device=device)
    return WaveFlowVocoder(spec, device=device)


def render(voc, sketch, controls=None, kind="event", seed=0, postprocess=True):
    """Controls (on the mel) + vocoder + cleanup. Returns (audio [T], edited sketch)."""
    from .controls import Controls, apply_controls
    from .generate import clean
    s = apply_controls(np.asarray(sketch, np.float32), controls or Controls(), kind, voc.mel_cfg, seed)
    a = voc(s, seed=seed)[0]
    return (clean(a, voc.sr, tone_removal=voc.kind == "waveflow") if postprocess else a), s
