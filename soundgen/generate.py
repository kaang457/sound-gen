"""Category + parameters -> audio.

    gen = Generator("runs/base/best.pt")
    out = gen.generate("footsteps", Controls(speed=1.4, weight=0.5, environment="room"), seed=3)

Where the sketch comes from:
  * by default, a reference clip of that label from the sketch bank (picked by seed)
  * or your own audio (`reference=`): style transfer, e.g. percussive -> explosion
Variation comes from: seed (reference choice + noise), temperature, MC dropout.
"""
from pathlib import Path

import numpy as np
import torch

from .audio import MelConfig, MelSketch, SR, save_audio, load_audio
from .controls import Controls, apply_controls
from .train import load_model


class Generator:
    def __init__(self, ckpt, bank=None, device=None, use_ema=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_model(ckpt, self.device, use_ema)
        self.vocab = ck["vocab"]
        self.mel_cfg = ck["mel_cfg"]
        self.mel = MelSketch(MelConfig(**self.mel_cfg)).to(self.device)
        bank = bank or Path(ckpt).parent / "sketch_bank.pt"
        self.bank = torch.load(bank, map_location="cpu", weights_only=False) if Path(bank).exists() else None
        self.step = ck.get("step")

    @property
    def labels(self):
        return self.vocab[1:]

    def _resolve(self, label):
        while label not in self.vocab and "/" in label:
            label = label.rsplit("/", 1)[0]
        if label not in self.vocab:
            raise KeyError(f"unknown label {label}; available: {self.labels}")
        return label

    def kind(self, label):
        types = (self.bank or {}).get("types", {})
        while label not in types and "/" in label:
            label = label.rsplit("/", 1)[0]
        return types.get(label, "event")

    def reference(self, label, seed=0, index=None):
        if not self.bank:
            raise RuntimeError("no sketch bank next to the checkpoint; pass reference= audio")
        items = self.bank["items"]
        while label not in items and "/" in label:
            label = label.rsplit("/", 1)[0]
        group = items[label]
        i = index if index is not None else int(np.random.default_rng(seed).integers(len(group)))
        return group[i % len(group)]

    def sketch_of(self, audio):
        x = torch.as_tensor(np.asarray(audio, np.float32))[None].to(self.device)
        return self.mel(x)[0].cpu().numpy()

    @torch.no_grad()
    def generate(self, label, controls=None, seed=0, temperature=0.6, mc_dropout=False,
                 reference=None, ref_index=None, n=1, postprocess=True, match=1.0):
        controls = controls or Controls()
        label_id = self.vocab.index(self._resolve(label))
        if reference is not None:
            if isinstance(reference, (str, Path)):
                reference = load_audio(reference, SR)
            base, ref = self.sketch_of(reference), dict(id="user", audio=None)
        else:
            ref = self.reference(label, seed, ref_index)
            base = ref["sketch"].float().numpy()
        sketch = apply_controls(base, controls, self.kind(label), self.mel_cfg, seed)
        torch.manual_seed(seed)
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.model.set_mc_dropout(mc_dropout)
        S = torch.from_numpy(sketch)[None].repeat(n, 1, 1).to(self.device)
        y = torch.full((n,), label_id, device=self.device, dtype=torch.long)
        audio = self.model.sample(S, y, temperature=temperature, generator=g).float().cpu().numpy()
        self.model.set_mc_dropout(False)
        if postprocess:
            audio = np.stack([clean(match_sketch(notch_tones(a, self.model.cfg.h), sketch, self.mel_cfg, match))
                              for a in audio])
        ref_audio = ref["audio"].numpy().astype(np.float32) / 32768 if ref.get("audio") is not None else None
        return dict(audio=audio, sketch=sketch, base_sketch=base, label=label, controls=controls.to_dict(),
                    seed=seed, temperature=temperature, mc_dropout=mc_dropout,
                    reference_id=ref.get("id"), reference_audio=ref_audio,
                    reference_public_ok=bool(ref.get("public_ok")), attribution=ref.get("attribution"))


def notch_tones(x, h, sr=SR, q=30.0):
    """Remove the constant tones at sr/h and harmonics (WaveFlow squeeze artefact)."""
    from scipy.signal import iirnotch, filtfilt
    x = np.asarray(x, np.float64)
    if len(x) < 64:
        return x
    f0 = sr / h
    for k in range(1, int((sr / 2) // f0) + 1):
        if k * f0 < sr / 2 - 1:
            b, a = iirnotch(k * f0, q, sr)
            x = filtfilt(b, a, x)
    return x


def match_sketch(x, sketch, mel_cfg, strength=1.0, max_boost_db=12.0, max_cut_db=40.0):
    """Pull the output's coarse (mel-band) spectrum toward the conditioning sketch.
    Boost is capped (amplifying a too-quiet output also amplifies its hiss); cuts are not."""
    if strength <= 0:
        return np.asarray(x, np.float64)
    import torchaudio
    n_fft, hop = mel_cfg["n_fft"], mel_cfg["hop"]
    fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, mel_cfg["f_min"], mel_cfg["f_max"],
                                               len(sketch), mel_cfg["sample_rate"],
                                               norm="slaney", mel_scale="slaney").numpy()
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    win = torch.hann_window(n_fft)
    X = torch.stft(t, n_fft, hop, window=win, return_complex=True, center=True)
    cur = fb.T @ (X.abs() ** 2).numpy()
    tgt = 10 ** ((np.asarray(sketch) * 100 - 100) / 10)
    n = min(cur.shape[1], tgt.shape[1])
    g = np.clip(10 * np.log10((tgt[:, :n] + 1e-10) / (cur[:, :n] + 1e-10)), -max_cut_db, max_boost_db) * strength
    g = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, g)
    w = fb / (fb.sum(1, keepdims=True) + 1e-9)
    gain = torch.from_numpy((10 ** ((w @ g) / 20)).astype(np.float32))
    y = torch.istft(X[:, :n] * gain, n_fft, hop, window=win, length=min(len(x), (n - 1) * hop))
    return y.numpy().astype(np.float64)


def clean(x, sr=SR, spike_db=12.0):
    """Post-processing: DC/rumble high-pass, narrow constant-tone removal
    (WaveFlow's known artefact, cf. Andreu & Villanueva 2022), short fades, soft limit."""
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import median_filter
    x = np.asarray(x, np.float64)
    if len(x) < 4096:
        return x.astype(np.float32)
    x = sosfiltfilt(butter(2, 25, "highpass", fs=sr, output="sos"), x)
    t = torch.from_numpy(np.ascontiguousarray(x))
    win = torch.hann_window(2048, dtype=torch.float64)
    X = torch.stft(t, 2048, 512, window=win, return_complex=True)
    avg = 20 * np.log10(X.abs().mean(dim=1).numpy() + 1e-9)
    excess = avg - median_filter(avg, size=31, mode="nearest")
    gain_db = np.where(excess > spike_db, -(excess - 3.0), 0.0)
    X = X * torch.from_numpy(10 ** (gain_db / 20))[:, None]
    x = torch.istft(X, 2048, 512, window=win, length=len(x)).numpy()
    f = int(0.005 * sr)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    peak = np.abs(x).max()
    if peak > 0.95:                                     # soft limiter, keeps relative levels
        x = np.tanh(x / peak * 2.0) / np.tanh(2.0) * 0.95
    return x.astype(np.float32)


def export(result, folder, name=None, sr=SR, max_s=6.0):
    """Write reference + outputs (+ json) - small enough for GitHub."""
    import json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = name or result["label"].replace("/", "__")
    n = int(max_s * sr)
    files = []
    for i, a in enumerate(result["audio"]):
        p = folder / f"{name}_gen{i}.wav"
        save_audio(p, a[:n], sr)
        files.append(p)
    if result.get("reference_audio") is not None:
        p = folder / f"{name}_ref.wav"
        save_audio(p, result["reference_audio"][:n], sr)
        files.append(p)
    meta = {k: v for k, v in result.items() if k not in ("audio", "sketch", "base_sketch", "reference_audio")}
    (folder / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    files.append(folder / f"{name}.json")
    return files
