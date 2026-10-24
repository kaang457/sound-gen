"""Prompt -> new sounds. One call = 3 variations + 5 blends between them = 8 candidates.

    g = Generator("runs/mm_v1/ema.pt", vae_ckpt)
    sounds = g.generate("gunfight in the rain", seconds=4, seed=0)                 # 8 dicts: name, kind, audio, sr
    sounds = g.generate("gunfight", prompt_b="thunderstorm", mix_b=0.3)            # two prompts mixed

A variation is one starting noise. A blend starts from a point BETWEEN the starting noises of two (or three)
variations and is generated anew by the model: one coherent new sound that inherits from both, not two sounds
layered. The sampler is deterministic (Euler, 25 steps), so the starting noise fully defines the sound.
Blends of two noises use spherical interpolation (keeps a valid Gaussian sample); the three-way centre is
(a + b + c) / sqrt(3) (independent Gaussians -> unit variance again).
"""
import math

import numpy as np
import torch

# (name, kind, weights over the 3 variations A, B, C); 2-way blends are slerped
PLAN = [("A", "varyasyon", (1, 0, 0)), ("B", "varyasyon", (0, 1, 0)), ("C", "varyasyon", (0, 0, 1)),
        ("A→B %33", "karışım", (2 / 3, 1 / 3, 0)), ("A→B %67", "karışım", (1 / 3, 2 / 3, 0)),
        ("B×C", "karışım", (0, 0.5, 0.5)), ("A×C", "karışım", (0.5, 0, 0.5)),
        ("A×B×C", "karışım", (1 / 3, 1 / 3, 1 / 3))]


def slerp(a, b, t):
    af, bf = a.flatten().double(), b.flatten().double()
    om = torch.arccos(torch.clamp((af @ bf) / (af.norm() * bf.norm()), -1, 1))
    if om.abs() < 1e-4:
        return (1 - t) * a + t * b
    return ((torch.sin((1 - t) * om) * a.double() + torch.sin(t * om) * b.double()) / torch.sin(om)).to(a.dtype)


def plan_noise(z, w):
    """z: [3, N, C] starting noises; w: weights over them -> one starting noise."""
    nz = [k for k, x in enumerate(w) if x > 0]
    if len(nz) == 1:
        return z[nz[0]]
    if len(nz) == 2:
        i, j = nz
        return slerp(z[i], z[j], w[j] / (w[i] + w[j]))
    v = sum(w[k] * z[k] for k in nz)
    return v / math.sqrt(sum(w[k] ** 2 for k in nz))


class Generator:
    def __init__(self, ckpt, vae_ckpt, device=None, steps=25, cfg=4.5, codec=None, encode_text=None):
        from mmaudio.model.flow_matching import FlowMatching
        from .train import build_network
        from .features import Codec, text_encoder
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        w = ck["weights"]
        self.net = build_network(ck["model"], w["latent_mean"].flatten(), w["latent_std"].flatten(),
                                 w["empty_string_feat"], **(ck.get("net_override") or {}))
        self.net.load_state_dict(w)
        self.net.eval().to(self.device)
        self.trained_lengths = (self.net._latent_seq_len, self.net._clip_seq_len, self.net._sync_seq_len)
        self.fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
        self.cfg, self.step = cfg, ck.get("step")
        self.codec = codec or Codec(vae_ckpt, self.device, need_encoder=False)
        self.encode_text = encode_text or text_encoder(self.device)
        self.sr = 44_100

    def text(self, prompt, prompt_b="", mix_b=0.0):
        f = self.encode_text([prompt])[0]
        if prompt_b and mix_b > 0:
            f = (1 - mix_b) * f + mix_b * self.encode_text([prompt_b])[0]
        return f.to(self.device)

    def _set_length(self, seconds):
        from mmaudio.model.sequence_config import SequenceConfig
        sc = SequenceConfig(duration=float(seconds), sampling_rate=44_100, spectrogram_frame_rate=512)
        self.net.update_seq_lengths(sc.latent_seq_len, sc.clip_seq_len, sc.sync_seq_len)
        return sc

    @torch.inference_mode()
    def generate(self, prompt, prompt_b="", mix_b=0.0, seconds=8.0, seed=0, plan=PLAN, cfg=None):
        sc = self._set_length(seconds)
        n_lat = sc.latent_seq_len
        g = torch.Generator().manual_seed(int(seed))
        z = torch.randn(3, n_lat, self.net.latent_dim, generator=g)
        x0 = torch.stack([plan_noise(z, w) for _, _, w in plan]).to(self.device)
        bs = len(x0)
        tf = self.text(prompt, prompt_b, mix_b)[None].expand(bs, -1, -1)
        clip_f = self.net.get_empty_clip_sequence(bs).contiguous()
        sync_f = self.net.get_empty_sync_sequence(bs).contiguous()
        cond = self.net.preprocess_conditions(clip_f, sync_f, tf.contiguous())
        empty = self.net.get_empty_conditions(bs)
        amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=amp):
            fn = lambda t, x: self.net.ode_wrapper(t, x, cond, empty, self.cfg if cfg is None else cfg)
            x1 = self.fm.to_data(fn, x0)
        lat = self.net.unnormalize(x1.float())
        wav = self.codec.decode(lat).float().cpu().numpy()
        n = int(seconds * self.sr)
        self.net.update_seq_lengths(*self.trained_lengths)               # back to the trained length
        out = []
        for (name, kind, _), a in zip(plan, wav):
            a = a[:n]
            f = min(len(a), int(0.01 * self.sr))
            a[-f:] *= np.linspace(1, 0, f)
            peak = np.abs(a).max()
            if peak > 0.98:
                a = a / peak * 0.98
            out.append(dict(name=name, kind=kind, audio=a.astype(np.float32), sr=self.sr, prompt=prompt,
                            prompt_b=prompt_b, mix_b=mix_b, seed=int(seed)))
        return out
