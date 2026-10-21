"""Stage 1: a small diffusion model that invents NEW sketches (the coarse 20-band "skeleton" of a sound).

    vector (category embedding, a mix of them, or any vector you type) -> new sketch -> controls -> WaveFlow -> audio

No reference sound is needed: the timing, number of events and spectral shape come out of noise, guided by
the vector. WaveFlow (stage 2) then adds the fine detail.

Design: DDPM (1000 steps, cosine schedule, epsilon prediction) over sketches of a fixed length, a 1-D U-Net
over time with the 20 bands as channels, FiLM conditioning on (diffusion step + vector), classifier-free
guidance (the vector is dropped 10 % of the time). Sampling: DDIM, 50 steps.
"""
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
