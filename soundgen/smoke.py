"""End-to-end smoke test on synthetic sounds (CPU or GPU, ~2-5 min on CPU).

    python -m soundgen.smoke

Checks: flow invertibility, exact log-det on a tiny model, pack building, base
training, transfer-learning fine-tune of a new sub-category, sketch bank,
parametric generation with MC dropout, export.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from .audio import SR, save_audio
from .controls import Controls
from .model import FlowConfig, WaveFlow


def check_flow(device="cpu"):
    for up in ("transposed", "interp"):
        _check_flow(device, up)
    _check_flow(device, "interp", fixes=True)


def _check_flow(device, upsampler, fixes=False):
    torch.manual_seed(0)
    cfg = FlowConfig(n_flows=4, n_layers=4, h=8, res_channels=16, n_mels=20, label_dim=8, dropout=0.1,
                     upsampler=upsampler, cond_norm=fixes, actnorm=fixes)
    m = WaveFlow(cfg, 3).to(device)
    if fixes:                                        # ActNorm's data-dependent init needs one train forward
        with torch.no_grad():
            m.train()(0.3 * torch.randn(2, 512, device=device), torch.rand(2, 20, 4, device=device),
                      torch.tensor([1, 2], device=device))
    m.eval()
    for f in m.flows:                                # non-trivial flows (end conv starts at zero)
        torch.nn.init.normal_(f.end.weight, std=0.05)
    T = 8 * 64
    x = 0.3 * torch.randn(2, T, device=device)
    s = torch.rand(2, 20, T // 256 + 2, device=device)
    y = torch.tensor([1, 2], device=device)
    z, logdet = m(x, s, y)
    x2 = m.reverse(z, s, y)
    err = float((x - x2).abs().max())
    assert err < 1e-4, f"inverse error {err}"
    # exact log-det vs autograd Jacobian on a tiny input
    xs = 0.3 * torch.randn(1, 8 * 4, device=device, dtype=torch.float64)
    m64 = m.double()
    s1 = torch.rand(1, 20, 2, device=device, dtype=torch.float64)
    f = lambda v: m64(v.view(1, -1), s1, y[:1])[0].transpose(2, 3).reshape(-1)
    J = torch.autograd.functional.jacobian(f, xs.view(-1))
    ld_true = torch.linalg.slogdet(J)[1].item()
    ld = m64(xs, s1, y[:1])[1].item()
    assert abs(ld - ld_true) < 1e-6 * max(1, abs(ld_true)) + 1e-6, f"logdet {ld} vs {ld_true}"
    m.float()
    print(f"flow ok ({upsampler} upsampler{', cond_norm + actnorm' if fixes else ''}): inverse err {err:.2e}, logdet {ld:.4f} == {ld_true:.4f}")


def synthetic_items(folder, n=40):
    """Three toy 'categories': clicks (sequence), noise bursts (event), hum (texture)."""
    rng = np.random.default_rng(0)
    folder = Path(folder)
    items = []
    for label in ("clicks", "burst", "hum", "clicks/soft"):
        for i in range(n if "/" not in label else 12):
            t = np.arange(int(1.5 * SR)) / SR
            if label.startswith("clicks"):
                x = np.zeros_like(t)
                for o in np.arange(0.05, 1.4, rng.uniform(0.2, 0.4)):
                    k = int(o * SR)
                    x[k:k + 400] += rng.normal(0, 0.5 if label == "clicks" else 0.15, 400) * np.exp(-np.arange(400) / 80)
            elif label == "burst":
                x = rng.normal(0, 0.5, len(t)) * np.exp(-t / rng.uniform(0.05, 0.3))
            else:
                x = 0.3 * np.sin(2 * math.pi * rng.uniform(80, 200) * t) + 0.02 * rng.normal(size=len(t))
            p = folder / label / f"{i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            save_audio(p, x)
            items.append(dict(path=str(p), label=label, source="synthetic", src_id=f"{label}{i}",
                              license="none", attribution="synthetic", public_ok=True))
    return items


def main(device=None):
    from .data import build_pack
    from .train import TrainConfig, train
    from .generate import Generator, export
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    check_flow(device)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        items = synthetic_items(tmp / "wav")
        build_pack(items, tmp / "pack")
        small = dict(n_flows=2, n_layers=4, h=8, res_channels=16, label_dim=16, upsampler="interp")
        base = TrainConfig(packs=[str(tmp / "pack")], out_dir=str(tmp / "base"), model=small, steps=30,
                           batch_size=4, segment=8192, warmup=5, eval_every=15, sample_every=30,
                           num_workers=0, min_clips=5, log_every=10)
        train(base, device)
        ft = TrainConfig(packs=[str(tmp / "pack")], out_dir=str(tmp / "ft"), labels=["clicks/soft"], steps=20,
                         batch_size=4, segment=8192, warmup=2, eval_every=10, sample_every=20, num_workers=0,
                         init_from=str(tmp / "base" / "best.pt"), freeze="last_flows", trainable_flows=1,
                         l2sp=1e-3, replay_labels=["clicks"], replay_ratio=0.3, min_clips=1, log_every=10)
        train(ft, device)
        gen = Generator(tmp / "ft" / "best.pt", device=device)
        assert "clicks/soft" in gen.labels and "hum" in gen.labels, gen.labels
        for label, c in [("clicks/soft", Controls(speed=1.5, duration=2.0, environment="room")),
                         ("burst", Controls(intensity=0.8, distance=0.5)),
                         ("hum", Controls(duration=3.0))]:
            out = gen.generate(label, c, seed=1, mc_dropout=True, n=2)
            a = out["audio"]
            assert np.isfinite(a).all() and a.shape[0] == 2, label
            export(out, tmp / "export")
            print(f"generated {label}: {a.shape[1] / SR:.2f}s, peak {np.abs(a).max():.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
