"""End-to-end test on synthetic data: python -m soundgen.smoke

Always: pack building (48 kHz), effects rack, copy detection (an exact copy scores 0 dB), event counting.
With MMAudio available (soundgen.mm.setup.ensure_mmaudio + ext_weights): features -> a tiny network trained for a
few steps -> 8 candidates. The text encoder is replaced by a deterministic stand-in here (CLIP is a 4 GB download);
the Colab notebook's dry run uses the real one.
"""
import hashlib
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .audio import save_audio

SR = 48000


def synthetic_items(folder, n=40):
    """Three toy 'categories': clicks (sequence), noise bursts (event), hum (texture)."""
    rng = np.random.default_rng(0)
    folder = Path(folder)
    items = []
    for label in ("clicks", "burst", "hum"):
        for i in range(n):
            t = np.arange(int(1.5 * SR)) / SR
            if label == "clicks":
                x = np.zeros_like(t)
                for o in np.arange(0.05, 1.4, rng.uniform(0.2, 0.4)):
                    k = int(o * SR)
                    x[k:k + 400] += rng.normal(0, 0.5, 400) * np.exp(-np.arange(400) / 80)
            elif label == "burst":
                x = rng.normal(0, 0.5, len(t)) * np.exp(-t / rng.uniform(0.05, 0.3))
            else:
                x = 0.3 * np.sin(2 * math.pi * rng.uniform(80, 200) * t) + 0.02 * rng.normal(size=len(t))
            p = folder / label / f"{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            save_audio(p, x, SR)
            items.append(dict(path=str(p), label=label, source="synthetic", src_id=f"{label}{i}",
                              license="none", attribution="synthetic", public_ok=True))
    return items


def fake_text(texts, batch=64):
    out = []
    for t in texts:
        g = torch.Generator().manual_seed(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        x = torch.randn(77, 1024, generator=g)
        out.append(x / x.norm(dim=-1, keepdim=True))
    return torch.stack(out)


TINY = dict(latent_dim=40, clip_dim=1024, sync_dim=768, text_dim=1024, hidden_dim=64, depth=2, fused_depth=1,
            num_heads=2, latent_seq_len=345, clip_seq_len=64, sync_seq_len=192)


def main(mm=None):
    from .data import build_pack
    from .fx import Rack, process
    from .checks import NoveltyIndex, count_events
    from .audio import MEL_V3
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_pack(synthetic_items(tmp / "wav", n=12), tmp / "pack", sr=SR)
        # effects rack
        x = np.zeros(44100, np.float32)
        x[1000:1400] = np.random.default_rng(0).normal(0, 0.5, 400)
        st, mo = process(x, 44100, Rack(intensity_db=-3, distance=0.5, environment="hall", pan=-0.4, stretch=1.2))
        assert st.shape[0] == 2 and np.isfinite(st).all() and np.abs(st).max() <= 1.0
        print(f"fx ok: stereo {st.shape}, peak {np.abs(st).max():.2f}")
        # copy detection + event counting
        ix = NoveltyIndex([str(tmp / "pack")], MEL_V3.to_dict(), 48, device="cpu")
        k = int((ix.owner == 3).nonzero()[0])
        q = ((ix.bank[:, k + 5: k + 53] + 100) / 100).numpy()
        d = ix.nearest(q)[0]
        assert d < 1e-3, d
        clicks = np.zeros(3 * 44100, np.float32)
        for t in (0.2, 0.9, 1.6, 2.5):
            clicks[int(t * 44100): int(t * 44100) + 400] = 0.5
        assert count_events(clicks, 44100) == 4
        print(f"checks ok: copy -> {d:.3f} dB, baseline {ix.baseline():.1f} dB, events 4")
        if mm is None:
            try:
                import mmaudio  # noqa: F401
                mm = True
            except ImportError:
                mm = False
        if mm:
            from .mm.setup import ext_weights
            from .mm.features import extract, Codec
            from .mm.train import train
            from .mm.generate import Generator
            w = ext_weights()
            codec = Codec(w["vae"], "cpu")
            extract([str(tmp / "pack")], tmp / "feat", tmp / "raw", w["vae"], device="cpu", batch=4,
                    num_workers=0, codec=codec, encode_text=fake_text)
            ck = train(tmp / "feat", tmp / "run", batch=4, steps=12, warmup=4, val_every=6, save_every=12,
                       log_every=6, empty_string=w["empty"], device="cpu", net_override=TINY)
            g = Generator(ck, w["vae"], device="cpu", steps=3, codec=codec, encode_text=fake_text)
            s = g.generate("a gunshot", prompt_b="thunder", mix_b=0.3, seconds=1.0, seed=1)
            assert len(s) == 8 and all(np.isfinite(x["audio"]).all() for x in s)
            print(f"mm ok: 8 candidates {[x['name'] for x in s]}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main(mm=("--mm" in sys.argv) or None)
