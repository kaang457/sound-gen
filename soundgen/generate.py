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
        self.sr = int(self.mel_cfg["sample_rate"])     # 22050 for poc_v1, 48000 for contract v2
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
    def generate(self, label, controls=None, seed=0, temperature=1.0, mc_dropout=False,
                 reference=None, ref_index=None, n=1, postprocess=True, refine=None):
        """refine="v1": comb notch at k*sr/16 + level/spectrum matched to the (edited) sketch, +-30 dB.
        Kept after listening tests (samples/base_v1/postprocess_v1); "v2" was rejected and removed."""
        controls = controls or Controls()
        label_id = self.vocab.index(self._resolve(label))
        if reference is not None:
            if isinstance(reference, (str, Path)):
                reference = load_audio(reference, self.sr)
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
            audio = np.stack([clean(a, self.sr) for a in audio])
        if refine == "v1":
            audio = np.stack([refine_v1(a, sketch, self.mel_cfg, sr=self.sr) for a in audio])
        elif refine:
            raise ValueError(f"unknown refine mode {refine!r} (use None or 'v1')")
        ref_audio = ref["audio"].numpy().astype(np.float32) / 32768 if ref.get("audio") is not None else None
        return dict(audio=audio, sketch=sketch, base_sketch=base, label=label, controls=controls.to_dict(),
                    seed=seed, temperature=temperature, mc_dropout=mc_dropout,
                    reference_id=ref.get("id"), reference_audio=ref_audio,
                    sr=self.sr, reference_public_ok=bool(ref.get("public_ok")), attribution=ref.get("attribution"))


def refine_v1(x, sketch, mel_cfg, sr=SR, max_db=30.0, q=30.0):
    """Post-processing v1 (listening test: kept). 1) notch the tones at k * sr/16 that the 16-row squeeze
    leaves; 2) push the output's coarse spectrum toward the conditioning sketch (+-max_db per band/frame).
    Note: step 2 imposes the sketch with DSP; compare with the same shaping on white noise to judge the model."""
    import torchaudio
    from scipy.signal import filtfilt, iirnotch
    x = np.asarray(x, np.float64)
    f0 = sr / 16
    for k in range(1, int((sr / 2) // f0) + 1):
        if k * f0 >= sr / 2 - 1:
            break
        b, a = iirnotch(k * f0, q, sr)
        x = filtfilt(b, a, x)
    c = mel_cfg if isinstance(mel_cfg, dict) else mel_cfg.to_dict()
    n_fft, hop = c["n_fft"], c["hop"]
    fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, c["f_min"], c["f_max"], c["n_mels"], sr,
                                               norm="slaney", mel_scale="slaney").numpy()      # [bins, bands]
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    win = torch.hann_window(n_fft)
    X = torch.stft(t, n_fft, hop, window=win, return_complex=True, center=True)
    cur = fb.T @ (X.abs() ** 2).numpy()
    tgt = 10 ** ((np.asarray(sketch) * 100 - 100) / 10)                  # sketch -> mel power
    F = min(cur.shape[1], tgt.shape[1])
    g = np.clip(10 * np.log10((tgt[:, :F] + 1e-10) / (cur[:, :F] + 1e-10)), -max_db, max_db)
    g = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, g)
    w = fb / (fb.sum(1, keepdims=True) + 1e-9)
    gain = torch.from_numpy((10 ** ((w @ g) / 20)).astype(np.float32))
    y = torch.istft(X[:, :F] * gain, n_fft, hop, window=win, length=min(len(x), (F - 1) * hop)).numpy()
    peak = np.abs(y).max()
    if peak > 0.95:
        y = y / peak * 0.95
    return y.astype(np.float32)


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


def export(result, folder, name=None, sr=None, max_s=6.0):
    """Write reference + outputs (+ json) - small enough for GitHub."""
    import json
    sr = sr or result.get("sr", SR)
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
