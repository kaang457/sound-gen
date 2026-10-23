"""Sound studio: prompt -> new sounds from scratch -> DAW-style effects -> WAV. Engine (no UI code here).

    st = Studio("runs/melgen_v1/mel_gen.pt")              # + BigVGAN-v2, both on the GPU
    sounds = st.generate("laser gunshot", n=4)            # 4 variations + 6 blends between them
    stereo, mono = st.process(sounds[0], Rack(distance=0.4, environment="hall", pan=-0.3))

Variations differ only in their starting noise. A blend between variations A and B starts from the spherical
interpolation of their two noises (slerp keeps it a valid Gaussian sample), so the model generates ONE new sound
that inherits from both, not a crossfade of two sounds. A second prompt mixes in at the vector level.
"""
import itertools
import tempfile
from pathlib import Path

import numpy as np
import torch

from .controls import Controls
from .fx import Rack, process


def slerp(a, b, t):
    af, bf = a.flatten(), b.flatten()
    cos = torch.clamp((af @ bf) / (af.norm() * bf.norm()), -1.0, 1.0)
    om = torch.arccos(cos)
    if om.abs() < 1e-4:
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * om) * a + torch.sin(t * om) * b) / torch.sin(om)


class Studio:
    def __init__(self, mel_gen, vocoder="bigvgan", device=None, steps=50):
        from .sketch_model import load_generator
        from .vocoder import load_vocoder
        from .audio import mel_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mg, ck = load_generator(mel_gen, self.device)
        self.voc = load_vocoder(vocoder, self.device)
        if ck.get("mel_cfg") and mel_config(ck["mel_cfg"]) != mel_config(self.voc.mel_cfg):
            raise ValueError("mel generator and vocoder use different mel contracts")
        self.sr, self.steps = self.voc.sr, steps
        self.fps = self.sr / self.voc.mel_cfg["hop"]
        self.out = Path(tempfile.mkdtemp(prefix="studio_"))
        self.labels = [l for l in self.mg.vocab[1:] if l != "other"]

    # ------------------------------------------------------------------ generation
    def vector(self, prompt, prompt_b="", mix_b=0.0):
        def one(p):
            p = p.strip()
            if p in self.mg.vocab:
                return self.mg.vector(label=p)
            if hasattr(self.mg, "label_vecs"):              # CLAP model: any text
                return self.mg.vector(text=p)
            raise ValueError(f"'{p}' is not a label of this model; labels: {self.labels}")
        v = one(prompt)
        if prompt_b and prompt_b.strip() and mix_b > 0:
            v = (1 - mix_b) * v + mix_b * one(prompt_b)
            if hasattr(self.mg, "label_vecs"):
                v = v / v.norm().clamp_min(1e-6)
        return v

    @torch.no_grad()
    def generate(self, prompt, prompt_b="", mix_b=0.0, n=4, seconds=4.0, seed=0, guidance=3.0, blends=True):
        """-> list of sounds: dict(name, kind, audio [T] float32, sr, mel). Variations V1..Vn, then blends."""
        vec = self.vector(prompt, prompt_b, mix_b).to(self.device)
        frames = max(8, int(round(seconds * self.fps / 4)) * 4)
        shape = (self.mg.cfg.n_mels, frames)
        z = [torch.randn(*shape, generator=torch.Generator().manual_seed(seed * 1000 + i)) for i in range(n)]
        names = [f"V{i + 1}" for i in range(n)]
        kinds = ["varyasyon"] * n
        if blends and n > 1:
            for i, j in itertools.combinations(range(n), 2):
                z.append(slerp(z[i], z[j], 0.5))
                names.append(f"V{i + 1}×V{j + 1}")
                kinds.append("karışım")
        Z = torch.stack(z)
        mels = []
        for k in range(0, len(Z), 8):                              # GPU memory: 8 at a time
            mels.append(self.mg.sample(vec, steps=self.steps, guidance=guidance, noise=Z[k: k + 8]).cpu())
        mels = torch.cat(mels).numpy()
        from .vocoder import render
        sounds = []
        for name, kind, m in zip(names, kinds, mels):
            a, edited = render(self.voc, m, Controls(), "event", seed)
            sounds.append(dict(name=name, kind=kind, audio=a.astype(np.float32), sr=self.sr, mel=edited,
                               prompt=prompt, prompt_b=prompt_b, mix_b=mix_b, seed=seed))
        return sounds

    # ------------------------------------------------------------------ effects + export
    def process(self, sound, rack: Rack):
        return process(sound["audio"], sound["sr"], rack)

    def export(self, sound, rack: Rack, tag=""):
        import soundfile as sf
        stereo, mono = self.process(sound, rack)
        base = _slug(f"{sound['prompt']}_{sound['name']}{tag}")
        fm, fs = self.out / f"{base}_mono.wav", self.out / f"{base}_stereo.wav"
        sf.write(fm, mono, sound["sr"], subtype="PCM_24")
        sf.write(fs, stereo.T, sound["sr"], subtype="PCM_24")
        return fm, fs, stereo, mono

    def export_all(self, sounds, rack: Rack):
        import zipfile
        z = self.out / f"{_slug(sounds[0]['prompt'])}_hepsi.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for s in sounds:
                fm, fs, _, _ = self.export(s, rack)
                zf.write(fm, fm.name)
                zf.write(fs, fs.name)
        return z


def _slug(s):
    keep = "".join(ch if ch.isalnum() or ch in "-_×" else "_" for ch in s.strip().lower())
    return keep.replace("×", "x")[:80].strip("_") or "ses"
