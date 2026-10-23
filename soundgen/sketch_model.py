"""Stage 1: a small diffusion model that invents NEW sketches (the coarse 20-band "skeleton" of a sound).

    vector (category embedding, a mix of them, or any vector you type) -> new sketch -> controls -> WaveFlow -> audio

No reference sound is needed: the timing, number of events and spectral shape come out of noise, guided by
the vector. WaveFlow (stage 2) then adds the fine detail.

Design: DDPM (1000 steps, cosine schedule, epsilon prediction) over sketches of a fixed length, a 1-D U-Net
over time with the 20 bands as channels, FiLM conditioning on (diffusion step + vector), classifier-free
guidance (the vector is dropped 10 % of the time). Sampling: DDIM, 50 steps.
"""
import json
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MEAN, SCALE = 0.6, 0.3          # sketch values 0..1.2 -> roughly -2..2


@dataclass
class SketchConfig:
    n_mels: int = 128           # mel contract v2 (the CPU proof of concept used 20 bands / 176 frames)
    frames: int = 200           # 2.0 s at 100 frames/s (divisible by 4)
    channels: int = 128
    vec_dim: int = 64
    steps: int = 1000


def _ts_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Res(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 3, padding=2, dilation=2)
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(e)[:, :, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class SketchUNet(nn.Module):
    def __init__(self, cfg: SketchConfig, n_labels: int):
        super().__init__()
        self.cfg = cfg
        c, e = cfg.channels, 4 * cfg.channels
        self.embed = nn.Embedding(n_labels, cfg.vec_dim)          # index 0 = null (unconditional)
        nn.init.normal_(self.embed.weight, std=0.5)
        self.temb = nn.Sequential(nn.Linear(c, e), nn.SiLU(), nn.Linear(e, e))
        self.vemb = nn.Sequential(nn.Linear(cfg.vec_dim, e), nn.SiLU(), nn.Linear(e, e))
        self.inp = nn.Conv1d(cfg.n_mels, c, 3, padding=1)
        self.d1, self.d2 = Res(c, c, e), Res(c, 2 * c, e)
        self.d3 = Res(2 * c, 2 * c, e)
        self.mid1, self.mid2 = Res(2 * c, 2 * c, e), Res(2 * c, 2 * c, e)
        self.u3 = Res(4 * c, 2 * c, e)
        self.u2, self.u1 = Res(4 * c, c, e), Res(2 * c, c, e)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, cfg.n_mels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, vec):
        e = self.temb(_ts_embed(t, self.cfg.channels)) + self.vemb(vec)
        h1 = self.d1(self.inp(x), e)                          # T
        h2 = self.d2(F.avg_pool1d(h1, 2), e)                  # T/2
        h3 = self.d3(F.avg_pool1d(h2, 2), e)                  # T/4
        m = self.mid2(self.mid1(h3, e), e)
        u = self.u3(torch.cat([m, h3], 1), e)
        u = self.u2(torch.cat([F.interpolate(u, scale_factor=2), h2], 1), e)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h1], 1), e)
        return self.out(u)


class SketchDiffusion(nn.Module):
    def __init__(self, cfg: SketchConfig, vocab):
        super().__init__()
        self.cfg, self.vocab = cfg, list(vocab)
        self.net = SketchUNet(cfg, len(self.vocab))
        s = 0.008
        t = torch.linspace(0, 1, cfg.steps + 1)
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = (ab / ab[0]).clamp(1e-5, 1.0)
        self.register_buffer("abar", ab[1:].float())        # alpha-bar for t = 1..steps

    # ---------------------------------------------------------------- vectors
    def vector(self, label=None, mix=None):
        """label -> its embedding; mix={label: weight, ...} -> weighted blend; None -> null (unconditional)."""
        E = self.net.embed.weight
        if mix:
            w = torch.tensor([float(v) for v in mix.values()], device=E.device)
            ids = torch.tensor([self.vocab.index(k) for k in mix], device=E.device)
            return (w[:, None] * E[ids]).sum(0) / w.sum()
        return E[self.vocab.index(label) if label else 0]

    # ---------------------------------------------------------------- training
    def loss(self, x0, labels, p_uncond=0.1):
        B = x0.shape[0]
        t = torch.randint(0, self.cfg.steps, (B,), device=x0.device)
        ab = self.abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        labels = torch.where(torch.rand(B, device=x0.device) < p_uncond, torch.zeros_like(labels), labels)
        return F.mse_loss(self.net(xt, t, self.net.embed(labels)), eps)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, vec, n=1, steps=50, guidance=2.0, frames=None, generator=None):
        """vec [vec_dim] (or [n, vec_dim]) -> n new sketches [n, n_mels, frames] in sketch units (0..1.2)."""
        dev = self.abar.device
        frames = frames or self.cfg.frames
        frames = max(4, frames // 4 * 4)
        vec = vec.to(dev)
        vec = vec[None].expand(n, -1) if vec.dim() == 1 else vec
        null = self.net.embed.weight[0][None].expand(n, -1)
        x = torch.randn(n, self.cfg.n_mels, frames, device=dev, generator=generator)
        ts = torch.linspace(self.cfg.steps - 1, 0, steps, device=dev).long()
        for i, t in enumerate(ts):
            tt = t.repeat(n)
            e_c = self.net(x, tt, vec)
            e_u = self.net(x, tt, null) if guidance != 1.0 else e_c
            eps = e_u + guidance * (e_c - e_u)
            ab = self.abar[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-2.5, 2.5)
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps        # DDIM, eta = 0
        return (x * SCALE + MEAN).clamp(0.0, 1.2)

    # ---------------------------------------------------------------- io
    def save(self, path, extra=None):
        torch.save(dict(state=self.state_dict(), cfg=asdict(self.cfg), vocab=self.vocab, **(extra or {})), path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(SketchConfig(**ck["cfg"]), ck["vocab"])
        m.load_state_dict(ck["state"])
        return m.to(device).eval(), ck


# -------------------------------------------------------------------- data + training
def crops_from_bank(bank, vocab, frames, rng, n):
    """Random fixed-length crops from a sketch bank (mostly containing the loudest frame)."""
    pool = [(vocab.index(l), it["sketch"].float().numpy()) for l, items in bank["items"].items()
            if l in vocab for it in items]
    xs, ys = [], []
    for _ in range(n):
        y, s = pool[int(rng.integers(len(pool)))]
        T = s.shape[1]
        if T <= frames:
            c = np.zeros((s.shape[0], frames), np.float32)
            st = int(rng.integers(0, frames - T + 1))
            c[:, st: st + T] = s
        else:
            pk = int(np.argmax(s.mean(0)))
            if rng.random() < 0.7:
                lo, hi = max(0, pk - frames + 8), min(pk - 2, T - frames)
                st = int(rng.integers(lo, hi + 1)) if hi >= lo else int(rng.integers(0, T - frames + 1))
            else:
                st = int(rng.integers(0, T - frames + 1))
            c = s[:, st: st + frames]
        g = rng.uniform(-0.03, 0.03)                    # +-3 dB level jitter
        xs.append(np.clip(c + g * (c > 0.02), 0, 1.2))
        ys.append(y)
    x = (torch.from_numpy(np.stack(xs)) - MEAN) / SCALE
    return x.float(), torch.tensor(ys)


def train_on_bank(bank, steps=4000, batch=32, lr=2e-4, cfg=None, device="cpu", seed=0, log_every=250):
    from tqdm.auto import tqdm
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = ["<null>"] + sorted(bank["items"])
    cfg = cfg or SketchConfig()
    m = SketchDiffusion(cfg, vocab).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    ema = SketchDiffusion(cfg, vocab).to(device)
    ema.load_state_dict(m.state_dict())
    ema.requires_grad_(False)
    hist = []
    bar = tqdm(range(steps), desc="sketch model", mininterval=5)
    for step in bar:
        x, y = crops_from_bank(bank, vocab, cfg.frames, rng, batch)
        loss = m.loss(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (step + 1) / 200)
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), m.parameters()):
                e.lerp_(p, 1 - 0.999 if step > 500 else 0.5)
        hist.append(float(loss))
        if (step + 1) % log_every == 0:
            bar.set_postfix(loss=f"{np.mean(hist[-log_every:]):.4f}")
    return ema.eval(), hist


# -------------------------------------------------------------------- training on packs (Colab, mel contract v2)
def train_on_packs(packs, out_dir, steps=100_000, batch=64, lr=2e-4, cfg=None, mel_cfg=None, device=None,
                   seed=0, labels=None, balance_alpha=0.5, p_uncond=0.1, max_minutes=None, save_every=5000,
                   log_every=200, num_workers=2):
    """Train the mel generator on the same packs WaveFlow uses: crops -> mel (on the GPU) -> diffusion loss.
    Writes <out_dir>/mel_gen.pt (EMA weights, cfg, vocab, mel_cfg) and loss.csv; resumes from last.pt."""
    import csv
    import time
    from pathlib import Path
    from collections import Counter
    from tqdm.auto import tqdm
    from .audio import MelConfig, MelSketch
    from .data import PackDataset, read_index

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mel_cfg = mel_cfg or MelConfig()
    cfg = cfg or SketchConfig(n_mels=mel_cfg.n_mels)
    if cfg.n_mels != mel_cfg.n_mels:
        raise ValueError(f"SketchConfig.n_mels {cfg.n_mels} != mel contract {mel_cfg.n_mels}")
    for p in packs:
        meta = json.loads((Path(p) / "meta.json").read_text())
        if int(meta.get("sample_rate", 22050)) != mel_cfg.sample_rate:
            raise ValueError(f"{p} is {meta.get('sample_rate')} Hz, mel contract is {mel_cfg.sample_rate} Hz")
    counts = Counter(r["label"] for p in packs for r in read_index(p) if r["split"] == "train")
    vocab = ["<null>"] + sorted(labels or counts)
    seg = (cfg.frames - 1) * mel_cfg.hop                      # -> exactly cfg.frames mel frames (center=True)
    ds = PackDataset(packs, vocab, "train", seg, vocab[1:], augment=True)
    w = ds.sample_weights(balance_alpha)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=2_000_000, replacement=True)
    loader = torch.utils.data.DataLoader(ds, batch, sampler=sampler, num_workers=num_workers, drop_last=True,
                                         pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    mel = MelSketch(mel_cfg).to(device)
    m = SketchDiffusion(cfg, vocab).to(device)
    ema = SketchDiffusion(cfg, vocab).to(device)
    ema.load_state_dict(m.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}")
    print(f"mel generator: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params, "
          f"{len(vocab) - 1} labels, {len(ds)} clips, crop {seg / mel_cfg.sample_rate:.2f} s")

    def save(final=False):
        torch.save(dict(model=m.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step), out / "last.pt")
        ema.save(out / "mel_gen.pt", extra=dict(mel_cfg=mel_cfg.to_dict(), step=step,
                                                 trained_on=[str(p) for p in packs]))
        if final:
            print(f"saved {out / 'mel_gen.pt'} (step {step})")

    log = open(out / "loss.csv", "a", newline="")
    wr = csv.writer(log)
    if step == 0:
        wr.writerow(["step", "loss", "lr", "sec_per_step"])
    t0, hist, it = time.time(), [], iter(loader)
    bar = tqdm(total=steps, initial=step, desc="mel generator", mininterval=10)
    m.train()
    while step < steps:
        try:
            x, y = next(it)
        except StopIteration:              # the sampler is finite (2M draws): start a new pass
            it = iter(loader)
            x, y = next(it)
        with torch.no_grad():
            s = mel(x.to(device, non_blocking=True))[:, :, : cfg.frames]
            x0 = (s - MEAN) / SCALE
        loss = m.loss(x0, y.to(device), p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        cur = lr * min(1.0, (step + 1) / 1000)
        for g in opt.param_groups:
            g["lr"] = cur
        opt.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), m.parameters()):
                e.lerp_(p, 1 - 0.9995 if step > 2000 else 0.5)
        step += 1
        hist.append(loss.item())
        bar.update(1)
        if step % log_every == 0:
            sps = (time.time() - t0) / log_every
            t0 = time.time()
            wr.writerow([step, f"{np.mean(hist[-log_every:]):.5f}", f"{cur:.2e}", f"{sps:.3f}"])
            log.flush()
            bar.set_postfix(loss=f"{np.mean(hist[-log_every:]):.4f}")
        if step % save_every == 0:
            save()
        if max_minutes and (time.time() - bar.start_t) / 60 > max_minutes:
            print(f"time budget reached at step {step}")
            break
    bar.close()
    save(final=True)
    log.close()
    return ema.eval()


def novelty(gen_sketch, bank_sketches, hop=4, floor_db=-60.0):
    """Nearest-neighbour distance (mean |dB| over loud cells, any time offset) of a generated sketch to a set of
    real ones. Compare with the same distance between real sketches: similar values = new, not copied."""
    a = gen_sketch * 100 - 100
    best = 1e9
    T = a.shape[1]
    for b in bank_sketches:
        b = b * 100 - 100
        if b.shape[1] < T:
            b = np.pad(b, ((0, 0), (0, T - b.shape[1])), constant_values=-100)
        for off in range(0, b.shape[1] - T + 1, hop):
            w = b[:, off: off + T]
            mk = (a > floor_db) | (w > floor_db)
            if mk.any():
                best = min(best, float(np.abs(a - w)[mk].mean()))
    return best
