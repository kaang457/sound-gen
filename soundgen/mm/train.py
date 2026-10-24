"""Flow-matching training of the MMAudio network on our cached features (text-only for now).

    train(features_dir, out_dir, model="small_44k", batch=128, max_minutes=240)

Per step (same objective as MMAudio's runner.train_fn):
    x1 = VAE sample (mean + std * eps), normalised with the dataset's latent statistics
    t  = sigmoid(N(0, 1))               (log-normal time sampling)
    x0 = N(0, I);  xt = (1 - t) x0 + t x1;  target velocity = x1 - x0
    video features = the network's learned "empty" tokens (no video yet); text dropped to "" with p = 0.1 (CFG)
Everything (latents, text tokens) sits on the GPU, so there is no data loader to wait for.
Checkpoints: out_dir/last.pt (resume) and out_dir/ema.pt (generation weights + latent stats + config).
"""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def build_network(model, latent_mean=None, latent_std=None, empty_string_feat=None, **override):
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    if override:                                      # tiny networks for CPU tests
        return MMAudio(latent_mean=latent_mean, latent_std=latent_std, empty_string_feat=empty_string_feat,
                       **override)
    return get_my_mmaudio(model, latent_mean=latent_mean, latent_std=latent_std,
                          empty_string_feat=empty_string_feat)


def train(features_dir, out_dir, model="small_44k", batch=128, lr=1e-4, warmup=1000, steps=10**9,
          max_minutes=None, ema_decay=0.999, null_p=0.1, balance_alpha=0.5, val_every=2000, save_every=2000,
          log_every=100, empty_string=None, device=None, seed=0, net_override=None):
    from tqdm.auto import tqdm
    from .features import load
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    F = load(features_dir)
    rows = F["rows"]
    tr = np.array([k for k, r in enumerate(rows) if r["split"] == "train"])
    va = np.array([k for k, r in enumerate(rows) if r["split"] != "train"])[:256]
    if len(va) == 0:
        va = tr[:64]
    # everything on the device (fp16): ~ N x 345 x 40 x 2 x 2 bytes
    mean = torch.from_numpy(np.array(F["mean"])).to(device)
    std = torch.from_numpy(np.array(F["std"])).to(device)
    text = torch.from_numpy(F["text"]).to(device)
    cap = torch.from_numpy(F["cap_idx"]).long().to(device)
    s1 = torch.zeros(mean.shape[-1], dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    for c in torch.from_numpy(tr).to(device).split(4096):      # latent statistics, in chunks (GPU memory)
        x = mean[c].double()
        s1 += x.sum(dim=(0, 1))
        s2 += (x * x).sum(dim=(0, 1))
    cnt = len(tr) * mean.shape[1]
    lat_mean = (s1 / cnt).float()
    lat_std = (s2 / cnt - (s1 / cnt) ** 2).clamp_min(1e-8).sqrt().float()
    # class-balanced sampling (count^-alpha), so the big "other" bucket does not drown the named sounds
    labels = [rows[k]["label"] for k in tr]
    counts = {l: labels.count(l) for l in set(labels)}
    w = torch.tensor([counts[l] ** -balance_alpha for l in labels], dtype=torch.float, device=device)
    tr_t = torch.from_numpy(tr).to(device)

    empty = torch.load(empty_string, weights_only=True)[0] if empty_string else None
    net = build_network(model, lat_mean.cpu(), lat_std.cpu(), empty, **(net_override or {})).to(device)
    ema = build_network(model, lat_mean.cpu(), lat_std.cpu(), empty, **(net_override or {})).to(device)
    ema.load_state_dict(net.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.95),
                            weight_decay=1e-6)
    step = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}")
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"{model}: {n_params:.0f} M params | {len(tr)} train clips, {len(va)} val | batch {batch} | "
          f"{len(F['text'])} captions")
    amp = device.type == "cuda"
    g_val = torch.Generator(device=device).manual_seed(1234)
    vb = torch.from_numpy(va[:min(len(va), batch)]).to(device)
    v_eps = torch.randn(len(vb), 345, 40, generator=g_val, device=device)
    v_x0 = torch.randn(len(vb), 345, 40, generator=g_val, device=device)
    v_t = torch.sigmoid(torch.randn(len(vb), generator=g_val, device=device))

    def batch_loss(m, idx, eps, x0, t, drop=None):
        x1 = (mean[idx].float() + std[idx].float() * eps - lat_mean) / lat_std
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        bs = len(idx)
        tf = text[cap[idx]].float()
        if drop is not None:
            tf = torch.where(drop[:, None, None], m.empty_string_feat[None].float(), tf)
        clip_f = m.get_empty_clip_sequence(bs).contiguous()      # no video yet: the learned "empty" tokens
        sync_f = m.get_empty_sync_sequence(bs).contiguous()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            v = m(xt, clip_f, sync_f, tf, t)
        return (v.float() - (x1 - x0)).pow(2).mean()

    def save():
        torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step), out / "last.pt")
        torch.save(dict(weights=ema.state_dict(), model=model, net_override=net_override, step=step,
                        features=str(features_dir), captions=len(F["text"])), out / "ema.pt")

    log = open(out / "loss.csv", "a", newline="")
    wr = csv.writer(log)
    if step == 0:
        wr.writerow(["step", "train_loss", "val_loss", "lr", "sec_per_step"])
    t_start, t_log, hist = time.time(), time.time(), []
    bar = tqdm(total=None if steps >= 10**9 else steps, initial=step, desc="training", mininterval=10)
    net.train()
    while step < steps:
        frac = (time.time() - t_start) / 60 / max_minutes if max_minutes else step / steps
        cur = lr * min(1.0, (step + 1) / warmup) * (0.1 if frac > 0.85 else 1.0)
        for gp in opt.param_groups:
            gp["lr"] = cur
        idx = tr_t[torch.multinomial(w, batch, replacement=True)]
        eps = torch.randn(batch, 345, 40, device=device)
        x0 = torch.randn(batch, 345, 40, device=device)
        t = torch.sigmoid(torch.randn(batch, device=device))
        drop = torch.rand(batch, device=device) < null_p
        loss = batch_loss(net, idx, eps, x0, t, drop)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = ema_decay if step > warmup else 0.9
            for e, p in zip(ema.parameters(), net.parameters()):
                e.lerp_(p, 1 - d)
        step += 1
        hist.append(loss.item())
        bar.update(1)
        vl = ""
        if step % val_every == 0:
            with torch.no_grad():
                ema.eval()
                vl = f"{batch_loss(ema, vb, v_eps, v_x0, v_t).item():.5f}"
        if step % log_every == 0 or vl:
            sps = (time.time() - t_log) / max(1, len(hist))
            wr.writerow([step, f"{np.mean(hist):.5f}", vl, f"{cur:.2e}", f"{sps:.3f}"])
            log.flush()
            bar.set_postfix(loss=f"{np.mean(hist):.4f}", val=vl or "-")
            hist, t_log = [], time.time()
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"loss is {loss.item()} at step {step}")
        if step % save_every == 0:
            save()
        if max_minutes and (time.time() - t_start) / 60 > max_minutes:
            print(f"time budget reached at step {step}")
            break
    bar.close()
    save()
    log.close()
    print(f"saved {out / 'ema.pt'} (step {step})")
    return out / "ema.pt"
