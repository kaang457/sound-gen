"""Sound studio: prompt -> 8 new sounds -> DAW-style effects -> WAV. Engine (no UI code here).

    st = Studio("runs/mm_v1/ema.pt", vae_ckpt)                      # our generator + MMAudio VAE + BigVGAN-v2
    sounds = st.generate("laser gunshot", seconds=4)               # 3 variations + 5 blends between them
    stereo, mono = st.process(sounds[0], Rack(distance=0.4, environment="hall", pan=-0.3))

How blends work and why they are new sounds (not crossfades): soundgen/mm/generate.py.
"""
import tempfile
from pathlib import Path

from .fx import Rack, process


class Studio:
    def __init__(self, ckpt, vae_ckpt, device=None, steps=25, cfg=4.5, generator=None):
        from .mm.generate import Generator
        self.gen = generator or Generator(ckpt, vae_ckpt, device=device, steps=steps, cfg=cfg)
        self.sr = self.gen.sr
        self.out = Path(tempfile.mkdtemp(prefix="studio_"))

    def generate(self, prompt, prompt_b="", mix_b=0.0, seconds=4.0, seed=0, guidance=None):
        return self.gen.generate(prompt, prompt_b, mix_b, seconds=seconds, seed=seed, cfg=guidance)

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
    s = s.replace("→", "to").replace("×", "x").replace("%", "")
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s.strip().lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep[:80].strip("_") or "ses"
