"""Conditional WaveFlow (Ping et al., ICML 2020) for multi-category Foley.

Additions on top of the paper, following Andreu & Villanueva (AIIDE 2022) and our
multi-category goal:
  * local conditioning by a LOW-resolution mel sketch (default 20 bands)
  * global conditioning by a label embedding (category or category/subcategory);
    index 0 is a "null" label, used for label dropout and for style transfer
  * dropout inside the gated layers (regularisation) that can be kept on at
    sampling time (Monte-Carlo dropout) as an extra source of variation

Shapes: audio [B, T] with T % h == 0; sketch [B, n_mels, F] with F >= T // hop.
"""
import math
from dataclasses import dataclass, asdict, field

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

LOG_2PI = math.log(2 * math.pi)
# Height dilations from WaveFlow Table 3 (filter size 3 over height).
HEIGHT_DILATIONS = {8: [1] * 8, 16: [1] * 8, 32: [1, 2, 4, 1, 2, 4, 1, 2], 64: [1, 2, 4, 8, 16, 1, 2, 4]}


@dataclass
class FlowConfig:
    n_flows: int = 8
    n_layers: int = 8
    h: int = 16                 # squeezed height = autoregressive steps at sampling
    res_channels: int = 64
    n_mels: int = 20
    hop: int = 256
    label_dim: int = 64
    dropout: float = 0.05
    height_dilations: list = field(default_factory=list)
    width_dilations: list = field(default_factory=list)
    # "transposed": WaveFlow paper (base_v1). "interp": linear interpolation after a 1-D conv; tests
    # hypothesis H1 (the stride-16 transposed convs cause the tones at k*22050/16 Hz). Default stays
    # "transposed" so checkpoints saved without this field load unchanged.
    upsampler: str = "transposed"
    # Fixes from the base_v1 health report (conditioning 10-45x weaker than the audio path). Off by default
    # so older checkpoints load unchanged.
    cond_norm: bool = False     # sketch (0..1.2, mean ~0.7) -> ~zero mean before the conditioner
    cond_scale: float = 0.25    # divisor after centring (0.25 -> about +-2.4; larger = gentler)
    actnorm: bool = False       # per-channel affine (data-initialised) on each flow's audio path

    def __post_init__(self):
        if not self.height_dilations:
            self.height_dilations = HEIGHT_DILATIONS.get(self.h, [1] * self.n_layers)[: self.n_layers]
        if not self.width_dilations:
            self.width_dilations = [2 ** i for i in range(self.n_layers)]
        rf = 2 * sum(self.height_dilations) + 1
        if rf < self.h:
            raise ValueError(f"height receptive field {rf} < h={self.h}")
        if self.hop % 16 or (self.hop // 16) % 2:
            raise ValueError("hop must be 16 * even number (e.g. 256)")

    def to_dict(self):
        return asdict(self)


class Upsampler(nn.Module):
    """Sketch frames -> sample rate with two transposed 2-D convs (WaveFlow 5.)."""

    def __init__(self, hop):
        super().__init__()
        s2 = hop // 16
        self.convs = nn.ModuleList([
            weight_norm(nn.ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))),
            weight_norm(nn.ConvTranspose2d(1, 1, (3, 2 * s2), stride=(1, s2), padding=(1, s2 // 2))),
        ])

    def forward(self, sketch):                      # [B, C, F] -> [B, C, F*hop]
        x = sketch.unsqueeze(1)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.4)
        return x.squeeze(1)


class InterpUpsampler(nn.Module):
    """Sketch frames -> sample rate by linear interpolation after a learnable 1-D conv.
    No pattern repeats every 16 samples, unlike stride-16 transposed convs, which the
    16-row squeeze would turn into tones at 22050/16 Hz and its harmonics."""

    def __init__(self, n_mels, hop):
        super().__init__()
        self.hop = hop
        self.conv = nn.Conv1d(n_mels, n_mels, 3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, sketch):                      # [B, C, F] -> [B, C, F*hop]
        x = sketch + self.conv(sketch)
        n = x.shape[-1]
        y = F.interpolate(x, size=(n - 1) * self.hop + 1, mode="linear", align_corners=True)
        return F.pad(y, (0, self.hop - 1), mode="replicate")   # frame k centred on sample k*hop


class ActNorm(nn.Module):
    """Glow's ActNorm on the coupling network's input features: a per-channel affine whose initial values
    make the first batch zero-mean / unit-variance. Purely affine afterwards, so causality is kept."""

    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def forward(self, h):
        if self.training and not bool(self.initialized):
            with torch.no_grad():
                hf = h.float()
                self.bias.copy_(-hf.mean(dim=(0, 2, 3), keepdim=True))
                self.logs.copy_(-torch.log(hf.std(dim=(0, 2, 3), keepdim=True) + 1e-6))
                self.initialized.fill_(1)
        return (h + self.bias.to(h.dtype)) * torch.exp(self.logs).to(h.dtype)


class WN2D(nn.Module):
    """One flow's network: predicts (log_s, t) for each squeezed sample from rows above it."""

    def __init__(self, cfg: FlowConfig):
        super().__init__()
        r, L = cfg.res_channels, cfg.n_layers
        self.r, self.L, self.p = r, L, cfg.dropout
        self.mc_dropout = False
        self.probe = None        # dict -> record activation statistics (diagnostics)
        self.ablate = None       # int -> silence that layer; "flow" -> identity flow (layer importance)
        self.start = weight_norm(nn.Conv2d(1, r, 1))
        self.an = ActNorm(r) if cfg.actnorm else None
        self.cond = weight_norm(nn.Conv2d(cfg.n_mels, 2 * r * L, 1))
        self.label = nn.Linear(cfg.label_dim, 2 * r * L)
        self.dil = list(zip(cfg.height_dilations, cfg.width_dilations))
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(r, 2 * r, 3, dilation=d)) for d in self.dil])
        self.res_skip = nn.ModuleList([
            weight_norm(nn.Conv2d(r, 2 * r if i < L - 1 else r, 1)) for i in range(L)])
        self.end = nn.Conv2d(r, 2, 1)
        nn.init.zeros_(self.end.weight)              # identity flow at start
        nn.init.zeros_(self.end.bias)

    def forward(self, x, c, g):
        if self.ablate == "flow":
            return torch.zeros_like(x), torch.zeros_like(x)
        probe = self.probe
        # Shift down one row: row i only sees rows < i (autoregressive over height).
        x = F.pad(x, (0, 0, 1, 0))[:, :, :-1]
        h = self.start(x)
        if self.an is not None:
            h = self.an(h)
        cond = self.cond(c) + self.label(g)[:, :, None, None]
        skip = 0
        drop = self.training or self.mc_dropout
        for i, ((dh, dw), conv, rs) in enumerate(zip(self.dil, self.convs, self.res_skip)):
            # causal over height (pad top only), non-causal over width
            y_audio = conv(F.pad(h, (dw, dw, 2 * dh, 0)))
            y_cond = cond[:, 2 * self.r * i: 2 * self.r * (i + 1)]
            y = y_audio + y_cond
            a, b = y.chunk(2, dim=1)
            acts = torch.tanh(a) * torch.sigmoid(b)
            if self.p > 0 and self.training:
                acts = F.dropout(acts, self.p, training=True)
            elif self.p > 0 and drop:
                # MC dropout at sampling: one mask per sample/channel/column, shared by
                # all rows, so the h autoregressive passes see the same sub-network.
                acts = acts * self._mask(i, acts)
            out = rs(acts)
            if i == self.ablate:
                out = torch.zeros_like(out)
            if probe is not None:
                _record_layer(probe, i, a, b, acts, h, out, self.r, last=i == self.L - 1,
                              y_audio=y_audio, y_cond=y_cond)
            if i < self.L - 1:
                h = (h + out[:, : self.r]) * math.sqrt(0.5)
                skip = skip + out[:, self.r:]
            else:
                skip = skip + out
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.end(F.relu(skip.to(self.end.weight.dtype)))
        log_s, t = out.chunk(2, dim=1)
        if probe is not None:
            _record_end(probe, skip, log_s, t, x)
        return log_s, t

    def reset_masks(self):
        self._masks = {}

    def _mask(self, i, acts):
        B, C, _, W = acts.shape
        m = getattr(self, "_masks", {}).get(i)
        if m is None or m.shape != (B, C, 1, W):
            keep = 1.0 - self.p
            m = torch.bernoulli(torch.full((B, C, 1, W), keep, device=acts.device)) / keep
            self._masks = {**getattr(self, "_masks", {}), i: m}
        return m.to(acts.dtype)


@torch.no_grad()
def _record_layer(probe, i, a, b, acts, h, out, r, last, y_audio=None, y_cond=None):
    """Activation health of one gated layer (diagnostics only; 0-dim tensors, no GPU sync)."""
    a, b, acts = a.float(), b.float(), acts.float()
    near_zero = (acts.abs() < 1e-3).float().mean(dim=(0, 2, 3))          # per channel
    stats = dict(
        tanh_sat=(a.abs() > 2.5).float().mean(),      # tanh' < 0.03: gradient barely passes
        gate_closed=(b < -4.0).float().mean(),        # sigmoid < 0.02: unit switched off
        dead_ch=(near_zero > 0.99).float().mean(),    # channels silent on (almost) every input
        skip=(out[:, r:] if not last else out).float().norm(),
    )
    if y_audio is not None:   # how loud the sketch/label is next to the audio path at the gate input
        stats["cond_ratio"] = y_cond.float().pow(2).mean().sqrt() / (y_audio.float().pow(2).mean().sqrt() + 1e-12)
    if not last:
        stats["res_ratio"] = out[:, :r].float().norm() / (h.float().norm() + 1e-12)
    probe.setdefault("layers", {})[i] = stats


@torch.no_grad()
def _record_end(probe, skip, log_s, t, x):
    s = skip.float()
    probe["end"] = dict(
        relu_dead=((s <= 0).float().mean(dim=(0, 2, 3)) > 0.99).float().mean(),
        log_s=log_s.float().abs().mean(), t=t.float().abs().mean(), x=x.float().pow(2).mean().sqrt())


def permute(x, k, n_flows):
    """WaveFlow permutation strategy (b); both are involutions."""
    if k < n_flows // 2:
        return x.flip(2)
    half = x.shape[2] // 2
    return torch.cat([x[:, :, :half].flip(2), x[:, :, half:].flip(2)], dim=2)


class WaveFlow(nn.Module):
    def __init__(self, cfg: FlowConfig, n_labels: int):
        super().__init__()
        self.cfg = cfg
        self.upsample = (InterpUpsampler(cfg.n_mels, cfg.hop) if cfg.upsampler == "interp"
                         else Upsampler(cfg.hop))
        self.embed = nn.Embedding(n_labels, cfg.label_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.flows = nn.ModuleList([WN2D(cfg) for _ in range(cfg.n_flows)])

    # ------------------------------------------------------------ helpers
    def set_mc_dropout(self, on: bool):
        for f in self.flows:
            f.mc_dropout = on

    def squeeze(self, a):                            # [B, C, T] -> [B, C, h, w]
        B, C, T = a.shape
        return a.view(B, C, T // self.cfg.h, self.cfg.h).transpose(2, 3)

    def unsqueeze(self, a):                          # [B, C, h, w] -> [B, C, T]
        B, C, h, w = a.shape
        return a.transpose(2, 3).reshape(B, C, h * w)

    def _conds(self, sketch, T):
        if self.cfg.cond_norm:
            sketch = (sketch - 0.6) / self.cfg.cond_scale
        c = self.upsample(sketch)
        if c.shape[-1] < T:
            raise ValueError(f"sketch too short: {c.shape[-1]} < {T} samples")
        c = self.squeeze(c[..., :T].contiguous())
        conds = []
        for k in range(self.cfg.n_flows):
            conds.append(c)
            c = permute(c, k, self.cfg.n_flows)
        return conds

    # ------------------------------------------------------------ density
    def forward(self, audio, sketch, labels):
        """x -> z. Returns z [B,1,h,w] and log|det J| [B]."""
        B, T = audio.shape
        if T % self.cfg.h:
            raise ValueError("audio length must be a multiple of h")
        g = self.embed(labels)
        conds = self._conds(sketch, T)
        x = self.squeeze(audio.unsqueeze(1))
        logdet = audio.new_zeros(B, dtype=self.flows[0].end.weight.dtype)
        for k, flow in enumerate(self.flows):
            log_s, t = flow(x, conds[k], g)
            x = x.to(log_s.dtype) * torch.exp(log_s) + t
            logdet = logdet + log_s.sum(dim=(1, 2, 3))
            x = permute(x, k, self.cfg.n_flows)
        return x, logdet

    def nll(self, audio, sketch, labels):
        """Negative log-likelihood in nats per audio sample (mean over batch)."""
        z, logdet = self(audio, sketch, labels)
        T = audio.shape[1]
        log_pz = (-0.5 * z ** 2 - 0.5 * LOG_2PI).sum(dim=(1, 2, 3))
        return -((log_pz + logdet) / T).mean()

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def reverse(self, z, sketch, labels):
        """z [B,1,h,w] -> audio [B, T]; h sequential steps per flow."""
        T = z.shape[2] * z.shape[3]
        g = self.embed(labels)
        conds = self._conds(sketch, T)
        x = z.to(self.embed.weight.dtype)
        for k in reversed(range(self.cfg.n_flows)):
            x = permute(x, k, self.cfg.n_flows)       # undo permutation after flow k
            zk, xp = x, torch.zeros_like(x)
            self.flows[k].reset_masks()
            for i in range(self.cfg.h):
                # row i depends only on rows < i (causal over height): feed rows 0..i only (~2x faster)
                log_s, t = self.flows[k](xp[:, :, : i + 1], conds[k][:, :, : i + 1], g)
                xp[:, :, i] = (zk[:, :, i] - t[:, :, i]) * torch.exp(-log_s[:, :, i])
            x = xp
        return self.unsqueeze(x).squeeze(1)

    @torch.no_grad()
    def sample(self, sketch, labels, temperature=1.0, generator=None):
        # temperature 1.0 = the level the model learned; 0.8 made poc_v1 outputs 7-9 dB too quiet
        B, _, n_frames = sketch.shape
        T = ((n_frames - 1) * self.cfg.hop) // self.cfg.h * self.cfg.h
        z = torch.randn(B, 1, self.cfg.h, T // self.cfg.h, device=sketch.device,
                        generator=generator) * temperature
        return self.reverse(z, sketch, labels)

    # ------------------------------------------------------------ transfer
    def grow_labels(self, n_new: int, init_from=None):
        """Append label rows; init each from a parent row (transfer learning)."""
        old = self.embed.weight.data
        emb = nn.Embedding(old.shape[0] + n_new, old.shape[1]).to(old.device)
        emb.weight.data[: old.shape[0]] = old
        for j in range(n_new):
            parent = (init_from or [None] * n_new)[j]
            base = old[parent] if parent is not None else old[0]
            emb.weight.data[old.shape[0] + j] = base + 0.01 * torch.randn_like(base)
        self.embed = emb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
