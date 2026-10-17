"""Training for the base model and transfer-learning fine-tunes.

Base:      all categories, label embedding, class-balanced sampling.
Fine-tune: start from a base checkpoint; new labels (e.g. footsteps/snow) get an
           embedding initialised from their parent (footsteps). Strategies:
             embedding_only - few clips (<~30): learn only the new label vector
             last_flows     - ~30-300 clips: + last k flows, earlier flows frozen
             full           - many clips: everything, low LR, L2-SP + replay
"""
import csv
from contextlib import nullcontext
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .audio import MelConfig, MelSketch, save_audio, SR
from .data import PackDataset, load_categories, read_index
from .model import FlowConfig, WaveFlow, count_parameters

NULL = "<null>"


@dataclass
class TrainConfig:
    packs: list
    out_dir: str
    labels: list = None                 # None: every label with >= min_clips
    min_clips: int = 30
    model: dict = field(default_factory=dict)   # FlowConfig overrides
    mel: dict = field(default_factory=dict)     # MelConfig overrides
    steps: int = 60000
    batch_size: int = 8
    segment: int = 16384                # ~0.74 s at 22.05 kHz, multiple of h
    lr: float = 2e-4
    min_lr: float = 1e-5
    warmup: int = 1000
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    label_dropout: float = 0.1          # -> null label; enables style transfer
    balance_alpha: float = 0.5          # 0 natural, 1 fully class-balanced
    sketch_noise: float = 0.02          # conditioner augmentation (robust to edited sketches)
    sketch_blur_p: float = 0.2
    amp: str = "auto"                   # auto | bf16 | off
    eval_every: int = 2000
    sample_every: int = 5000
    keep_steps: list = field(default_factory=lambda: [10000, 20000, 50000])
    num_workers: int = 2
    seed: int = 0
    # ---- transfer learning
    init_from: str = None               # base checkpoint
    freeze: str = "none"                # none | embedding_only | last_flows | full
    trainable_flows: int = 2
    embed_lr_mult: float = 10.0
    l2sp: float = 0.0                   # pull weights toward the base (anti-forgetting)
    replay_labels: list = None          # base labels mixed in during fine-tuning
    replay_ratio: float = 0.0
    log_every: int = 100
    max_minutes: float = None           # time budget: stop cleanly (eval + save + samples) when reached


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path, model, ema, opt, sched, step, cfg, vocab, best):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                    opt=opt.state_dict() if opt else None, sched=sched.state_dict() if sched else None,
                    step=step, flow_cfg=model.cfg.to_dict(), mel_cfg=cfg["mel_cfg"],
                    vocab=vocab, best=best, train_cfg=cfg["train_cfg"]), tmp)
    tmp.replace(path)


def load_model(path, device="cpu", use_ema=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    fcfg = FlowConfig(**ck["flow_cfg"])
    model = WaveFlow(fcfg, len(ck["vocab"]))
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    model.to(device).eval()
    return model, ck


class EMA:
    def __init__(self, model, decay):
        # Rebuild instead of deepcopy: weight-norm parametrisations do not always deepcopy.
        self.decay = decay
        self.model = WaveFlow(model.cfg, model.embed.num_embeddings).to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.model.state_dict()


# ------------------------------------------------------------------ helpers
def _amp_dtype(mode, device):
    if device.type != "cuda" or mode == "off":
        return None
    if mode in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None          # fp16 is unstable for flows' log-det; fall back to fp32


def _autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _augment_sketch(s, noise, blur_p):
    if noise > 0:
        s = s + noise * torch.randn_like(s)
    if blur_p > 0:
        m = (torch.rand(s.shape[0], 1, 1, device=s.device) < blur_p).float()
        s = m * F.avg_pool1d(s, 3, 1, 1, count_include_pad=False) + (1 - m) * s
    return s.clamp(0.0, 1.2)


def _cosine(step, warmup, total, lr, min_lr):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return (min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))) / lr


def _label_parent(label, vocab):
    parts = label.split("/")
    for k in range(len(parts) - 1, 0, -1):
        parent = "/".join(parts[:k])
        if parent in vocab:
            return vocab.index(parent)
    return 0


# ------------------------------------------------------------------ main loop
def train(cfg: TrainConfig, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(cfg.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # -------- labels / vocab
    all_rows = [r for p in cfg.packs for r in read_index(p) if r["split"] == "train"]
    counts = Counter(r["label"] for r in all_rows)
    # sub-categories (parent/child) are fine-tuning targets, not base labels, unless listed
    targets = cfg.labels or sorted(l for l, n in counts.items() if n >= cfg.min_clips and "/" not in l)
    missing = [l for l in targets if counts[l] == 0]
    if missing:
        raise ValueError(f"no training clips for {missing}; check the packs")
    base_ck = torch.load(cfg.init_from, map_location="cpu", weights_only=False) if cfg.init_from else None
    if base_ck:
        vocab = list(base_ck["vocab"])
        new = [l for l in targets if l not in vocab]
        mel_cfg = MelConfig(**base_ck["mel_cfg"])
        model = WaveFlow(FlowConfig(**base_ck["flow_cfg"]), len(vocab))
        model.load_state_dict(base_ck["ema"])            # transfer the EMA (smoother) weights
        parents = [_label_parent(l, vocab) for l in new]
        if new:
            model.grow_labels(len(new), parents)
            print("new labels:", {l: vocab[p] for l, p in zip(new, parents)})
        vocab += new
        n_old = len(base_ck["vocab"])
    else:
        vocab = [NULL] + targets
        mel_cfg = MelConfig(**cfg.mel)
        model = WaveFlow(FlowConfig(n_mels=mel_cfg.n_mels, hop=mel_cfg.hop, **cfg.model), len(vocab))
        n_old = 0
    model.to(device)
    mel = MelSketch(mel_cfg).to(device)
    print(f"model: {count_parameters(model) / 1e6:.2f} M params, labels: {len(vocab) - 1}")

    # -------- freezing (transfer learning)
    for p in model.parameters():
        p.requires_grad_(True)
    if base_ck and cfg.freeze in ("embedding_only", "last_flows"):
        model.requires_grad_(False)
        model.embed.requires_grad_(True)
        if cfg.freeze == "last_flows":
            for f in model.flows[-cfg.trainable_flows:]:
                f.requires_grad_(True)
        if n_old:   # keep the base labels' embeddings unchanged
            mask = torch.zeros(len(vocab), 1, device=device)
            mask[n_old:] = 1
            model.embed.weight.register_hook(lambda g: g * mask)
    trainable = [p for p in model.parameters() if p.requires_grad]
    anchor = [p.detach().clone() for p in trainable] if cfg.l2sp > 0 else None
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.2f} M ({cfg.freeze})")

    # -------- data
    seg = cfg.segment // model.cfg.h * model.cfg.h
    labels_used = list(targets) + list(cfg.replay_labels or [])
    ds = PackDataset(cfg.packs, vocab, "train", seg, labels_used, augment=True)
    w = ds.sample_weights(cfg.balance_alpha)
    if cfg.replay_labels and cfg.replay_ratio > 0:
        is_replay = np.array([r["label"] not in targets for r in ds.rows])
        if is_replay.any() and (~is_replay).any():
            w[is_replay] *= cfg.replay_ratio / w[is_replay].sum()
            w[~is_replay] *= (1 - cfg.replay_ratio) / w[~is_replay].sum()
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=min(cfg.steps * cfg.batch_size, 2_000_000), replacement=True)
    loader = torch.utils.data.DataLoader(ds, cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
                                         pin_memory=device.type == "cuda", drop_last=True,
                                         persistent_workers=cfg.num_workers > 0)
    batches = _infinite(loader)
    val_ds = PackDataset(cfg.packs, vocab, "val", seg, targets, augment=False)
    if len(val_ds) == 0:
        print("WARNING: no validation clips for these labels; validating on train clips")
        val_ds = PackDataset(cfg.packs, vocab, "train", seg, targets, augment=False)
    rng_state = np.random.get_state()
    np.random.seed(1234)
    val_items = [val_ds[i % len(val_ds)] for i in range(min(64, max(len(val_ds), 1) * 2))] if len(val_ds) else []
    np.random.set_state(rng_state)
    print(f"train clips: {len(ds)} {dict(ds.counts)}\nval crops: {len(val_items)}")

    # -------- optimiser
    embed_params = [p for p in model.embed.parameters() if p.requires_grad]
    other = [p for p in trainable if all(p is not q for q in embed_params)]
    groups = [dict(params=other, lr=cfg.lr)] if other else []
    groups.append(dict(params=embed_params, lr=cfg.lr * (cfg.embed_lr_mult if base_ck else 1.0)))
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _cosine(s, cfg.warmup, cfg.steps, cfg.lr, cfg.min_lr))
    ema = EMA(model, cfg.ema_decay)
    meta = dict(mel_cfg=mel_cfg.to_dict(), train_cfg=asdict(cfg))
    if not (out / "sketch_bank.pt").exists():      # data-only: build first so an early stop can still generate
        build_sketch_bank(cfg.packs, vocab, mel_cfg, out / "sketch_bank.pt")
    step, best = 0, float("inf")
    last = out / "last.pt"
    if last.exists():                                     # resume
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["vocab"] == vocab:
            model.load_state_dict(ck["model"])
            ema.model.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            step, best = ck["step"], ck["best"]
            print(f"resumed from step {step}")
    amp = _amp_dtype(cfg.amp, device)
    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    logger = csv.writer(log_f)
    if new_log:
        logger.writerow(["step", "train_nll", "val_nll", "grad_norm", "lr", "sec_per_step"])

    @torch.no_grad()
    def validate():
        if not val_items:
            return float("nan")
        m = ema.model
        total = 0.0
        for i in range(0, len(val_items), cfg.batch_size):
            xs, ys = zip(*val_items[i: i + cfg.batch_size])
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.tensor(ys, device=device)
            with _autocast(device, amp):
                total += float(m.nll(x, mel(x), y)) * len(xs)
        return total / len(val_items)

    model.train()
    t0, running, bad = time.time(), [], 0
    bar = tqdm(total=cfg.steps, initial=step, desc="training", unit="step", dynamic_ncols=True, mininterval=2)
    t_start = time.time()
    if cfg.max_minutes:
        tqdm.write(f"time budget: {cfg.max_minutes:.1f} min -> stops cleanly with a checkpoint and samples")
    last_val = best
    while step < cfg.steps:
        x, y = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = x + (torch.rand_like(x) - 0.5) / 32768.0          # dequantisation noise
        sketch = _augment_sketch(mel(x), cfg.sketch_noise, cfg.sketch_blur_p)
        if cfg.label_dropout > 0:
            y = torch.where(torch.rand_like(y, dtype=torch.float) < cfg.label_dropout, 0, y)
        with _autocast(device, amp):
            loss = model.nll(x, sketch, y)
        total = loss
        if anchor is not None:
            total = total + cfg.l2sp * sum(((p - a) ** 2).sum() for p, a in zip(trainable, anchor))
        if not torch.isfinite(total):
            bad += 1
            tqdm.write(f"step {step}: non-finite loss, batch skipped ({bad})")
            opt.zero_grad(set_to_none=True)
            if bad > 20:
                raise RuntimeError("loss keeps diverging: lower lr or check the data")
            continue
        bad = 0
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip))
        opt.step()
        sched.step()
        ema.update(model)
        step += 1
        bar.update(1)
        running.append(loss.item())

        if step % cfg.log_every == 0:
            sps = (time.time() - t0) / cfg.log_every
            bar.set_postfix(nll=f"{np.mean(running):.4f}", val=f"{last_val:.4f}", grad=f"{gnorm:.2f}",
                            lr=f"{sched.get_last_lr()[0]:.1e}")
            logger.writerow([step, np.mean(running), "", gnorm, sched.get_last_lr()[0], sps])
            log_f.flush()
            running, t0 = [], time.time()
        stop = bool(cfg.max_minutes) and (time.time() - t_start) / 60 >= cfg.max_minutes
        if stop:
            tqdm.write(f"time budget reached at step {step}")
        if step % cfg.eval_every == 0 or step == cfg.steps or stop:
            bar.set_description("validating")
            v = validate()
            last_val = v
            bar.set_description("training")
            logger.writerow([step, "", v, "", "", ""])
            log_f.flush()
            tqdm.write(f"step {step}: val nll {v:.4f} (best so far {min(best, v):.4f}), checkpoint saved")
            if v < best or (math.isnan(v) and not math.isfinite(best)):
                best = v if not math.isnan(v) else best
                save_checkpoint(out / "best.pt", model, ema, None, None, step, meta, vocab, best)
            save_checkpoint(last, model, ema, opt, sched, step, meta, vocab, best)
            if step in cfg.keep_steps:
                save_checkpoint(out / f"step_{step}.pt", model, ema, None, None, step, meta, vocab, best)
        if step % cfg.sample_every == 0 or step == cfg.steps or stop:
            bar.set_description("writing samples")
            _write_samples(ema.model, mel, val_ds, vocab, out / "samples" / f"step_{step:06d}", device,
                           fallback=ds)
            bar.set_description("training")
            tqdm.write(f"step {step}: samples -> {out / 'samples' / f'step_{step:06d}'}")
            model.train()
        if stop:
            break
    bar.close()
    log_f.close()
    if not (out / "sketch_bank.pt").exists():
        build_sketch_bank(cfg.packs, vocab, mel_cfg, out / "sketch_bank.pt")
    print("done. checkpoints in", out)
    return out


@torch.no_grad()
def _write_samples(model, mel, ds, vocab, folder, device, per_label=None, max_labels=None, seconds=3.0,
                   fallback=None):
    """Resynthesis check: real val sketch -> model -> audio, saved next to the reference.
    Every label gets a pair; labels without a val clip fall back to a train clip.
    per_label=None: about 6 pairs in total (a single-category model gets 6 of its own)."""
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    seen = Counter()
    n = int(seconds * SR) // model.cfg.h * model.cfg.h
    rows = list(ds.rows)
    if fallback is not None:
        have = {r["label"] for r in rows}
        rows += [r for r in fallback.rows if r["label"] not in have]
    labels = sorted({r["label"] for r in rows})[: max_labels or None]
    per_label = per_label or max(1, 6 // max(1, len(labels)))
    bar = tqdm(total=len(labels) * per_label, desc="samples", unit="clip", leave=False)
    for r in rows:
        if r["label"] not in labels or seen[r["label"]] >= per_label:
            continue
        a = ds.audio[r["pack"]][r["offset"]: r["offset"] + min(r["length"], n)].astype(np.float32) / 32768
        a = a[: len(a) // model.cfg.h * model.cfg.h]
        if len(a) < SR // 2:
            continue
        x = torch.from_numpy(a)[None].to(device)
        y = torch.tensor([vocab.index(r["label"])], device=device)
        gen = model.sample(mel(x), y, temperature=0.8)[0].float().cpu().numpy()
        name = r["label"].replace("/", "__") + (f"_{seen[r['label']]}" if per_label > 1 else "")
        save_audio(folder / f"{name}_ref.wav", a)
        save_audio(folder / f"{name}_gen.wav", gen)
        seen[r["label"]] += 1
        bar.update(1)
    bar.close()


def build_sketch_bank(packs, vocab, mel_cfg, path, per_label=48, ref_audio=4, max_s=6.0):
    """Reference sketches per label (used when generating without a user reference)."""
    cats = load_categories()["categories"]
    mel = MelSketch(mel_cfg)
    rng = np.random.default_rng(0)
    bank = dict(mel_cfg=mel_cfg.to_dict(), types={}, items={})
    rows = [r for p in packs for r in read_index(p) if r["split"] == "train"]
    audio = {p: np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r") for p in packs}
    for label in tqdm(vocab[1:], desc="sketch bank", unit="label"):
        group = [r for r in rows if r["label"] == label]
        if not group:
            continue
        rng.shuffle(group)
        items = []
        for k, r in enumerate(group[:per_label]):
            a = np.asarray(audio[r["pack"]][r["offset"]: r["offset"] + min(r["length"], int(max_s * SR))])
            s = mel(torch.from_numpy(a.astype(np.float32) / 32768)[None])[0].half()
            items.append(dict(sketch=s, id=r["id"], attribution=r["attribution"],
                              public_ok=r.get("public_ok") == "1",
                              audio=torch.from_numpy(a.copy()) if k < ref_audio else None))
        bank["items"][label] = items
        bank["types"][label] = cats.get(label, cats.get(label.split("/")[0], {})).get("type", "event")
    torch.save(bank, path)
    print(f"sketch bank: {len(bank['items'])} labels -> {path}")
    return bank
