"""Training health: dead neurons, vanishing/exploding gradients, unused layers, over/underfitting.

Two sources, one report:
  * during training  - Monitor writes <run>/diag.jsonl every `diag_every` steps:
                       per-layer gradient norm (before clipping), real update size
                       ||w_after - w_before|| / ||w_before||, clipping rate, activation health
  * from a checkpoint - diagnose_checkpoint(): gradients on a few batches, Adam's next update
                       (from the saved optimizer state), activation health and layer importance
                       (val NLL increase when one layer / flow is switched off)

report(run_dir, packs) combines both with log.csv into <run>/health/report.html and prints a
plain-text table. Every finding has a status (OK / UYARI / KRİTİK), the measured value, what it
means and what to do.

Why "update size" and not only gradient size: Adam divides each gradient by its own running
RMS, so a tiny gradient can still move a weight by ~lr. A layer is really stuck when its
weights stop moving (update ratio < 1e-5), or when its gradient is exactly zero.
"""
import base64
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------ thresholds (documented in the report)
T = dict(
    grad_dead=1e-12,          # gradient norm treated as exactly zero
    grad_vanish=1e-3,         # layer gradient < 0.1% of the median layer -> vanishing
    depth_warn=0.1,           # first flow / last flow gradient ratio
    depth_crit=0.01,
    upd_low=1e-5,             # update ratio: weights barely move
    upd_high=1e-1,            # update ratio: steps too large
    clip_warn=0.5,            # fraction of steps clipped
    dead_warn=0.10, dead_crit=0.30,      # fraction of dead channels in a layer
    sat_warn=0.30,            # fraction of saturated tanh inputs
    gate_warn=0.50,           # fraction of closed sigmoid gates
    relu_warn=0.25,           # dead channels before the last 1x1 conv
    unused_rel=0.01,          # layer importance < 1% of the median layer -> unused
)

OK, WARN, CRIT, INFO = "OK", "UYARI", "KRİTİK", "BİLGİ"
_ICON = {OK: "✓", WARN: "!", CRIT: "✕", INFO: "i"}


# ------------------------------------------------------------------ parameter groups
def group_of(name):
    """flows.3.convs.5.parametrizations.weight.original1 -> 'f3.L5'; flows.3.cond... -> 'f3.cond'."""
    m = re.match(r"flows\.(\d+)\.(convs|res_skip)\.(\d+)\.", name)
    if m:
        return f"f{m.group(1)}.L{m.group(3)}"
    m = re.match(r"flows\.(\d+)\.(\w+)", name)
    if m:
        return f"f{m.group(1)}.{m.group(2)}"
    return name.split(".")[0]                         # upsample, embed


def _group_norms(model, attr):
    sq = defaultdict(float)
    zero = defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        if not p.requires_grad or name.endswith(".bias"):
            continue      # weights only: a bias keeps getting gradient even when every unit is dead
        t = p.grad if attr == "grad" else p.detach()
        if t is None:
            continue
        g = group_of(name)
        sq[g] += float(t.float().pow(2).sum())
        if attr == "grad":
            zero[g][0] += int((t == 0).sum())
            zero[g][1] += t.numel()
    norms = {g: math.sqrt(v) for g, v in sq.items()}
    return norms, {g: z / max(n, 1) for g, (z, n) in zero.items()}


def _collect_probes(model):
    layers, flows = {}, {}
    for k, f in enumerate(model.flows):
        pr = f.probe or {}
        for i, st in pr.get("layers", {}).items():
            layers[f"f{k}.L{i}"] = {n: float(v) for n, v in st.items()}
        if "end" in pr:
            flows[f"f{k}"] = {n: float(v) for n, v in pr["end"].items()}
    return layers, flows


def set_probes(model, on):
    for f in model.flows:
        f.probe = {} if on else None


# ------------------------------------------------------------------ during training
class Monitor:
    """Usage in the training loop (see train.py):
        if mon.due(step): mon.start(model)        before forward
        ... backward ...
        if mon.active: mon.grads(model)           before clip_grad_norm_
        mon.count(gnorm)                          every step
        if mon.active: mon.snapshot(model)        before opt.step()
        if mon.active: mon.finish(model, step, loss, gnorm, lr)   after opt.step()
    """

    def __init__(self, out_dir, every=250, clip=1.0):
        self.path = Path(out_dir) / "diag.jsonl"
        self.every, self.clip = every, clip
        self.active = False
        self.n_steps = self.n_clipped = self.n_nonfinite = 0

    def due(self, step):
        return self.every > 0 and (step % self.every == 0 or step in (1, 50))

    def start(self, model):
        self.active = True
        set_probes(model, True)

    def grads(self, model):
        self.g, self.zero = _group_norms(model, "grad")

    def count(self, gnorm):
        self.n_steps += 1
        if not math.isfinite(gnorm):
            self.n_nonfinite += 1
        elif gnorm > self.clip:
            self.n_clipped += 1

    @torch.no_grad()
    def snapshot(self, model):
        self.before = {n: p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad and not n.endswith(".bias")}

    @torch.no_grad()
    def finish(self, model, step, loss, gnorm, lr):
        dsq, wsq = defaultdict(float), defaultdict(float)
        for n, p in model.named_parameters():
            if n in self.before:
                g = group_of(n)
                dsq[g] += float((p.detach() - self.before[n]).float().pow(2).sum())
                wsq[g] += float(self.before[n].float().pow(2).sum())
        layers, flows = _collect_probes(model)
        set_probes(model, False)
        rec = dict(step=step, loss=float(loss), gnorm=float(gnorm), lr=float(lr),
                   clip_frac=self.n_clipped / max(self.n_steps, 1),
                   nonfinite=self.n_nonfinite, steps=self.n_steps,
                   groups={g: dict(g=self.g.get(g, 0.0), zero=self.zero.get(g, 0.0),
                                   u=math.sqrt(dsq[g]) / (math.sqrt(wsq[g]) + 1e-12))
                           for g in dsq},
                   layers=layers, flows=flows)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.before, self.active = None, False
        self.n_steps = self.n_clipped = self.n_nonfinite = 0


def read_history(run_dir):
    p = Path(run_dir) / "diag.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ from a checkpoint
def _batches(packs, vocab, n, batch, segment, seed=0, split="val"):
    from .data import PackDataset
    ds = PackDataset(packs, vocab, split, segment, augment=False)
    if len(ds) == 0:
        ds = PackDataset(packs, vocab, "train", segment, augment=False)
    rng = np.random.default_rng(seed)
    state = np.random.get_state()
    np.random.seed(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(len(ds), size=batch)
        xs, ys = zip(*[ds[int(i)] for i in idx])
        out.append((torch.from_numpy(np.stack(xs)), torch.tensor(ys)))
    np.random.set_state(state)
    return out


def _adam_update_ratio(model, ck):
    """Size of Adam's next step per group, from the saved optimizer state (needs last.pt)."""
    opt = ck.get("opt")
    if not opt:
        return {}
    named = list(model.named_parameters())
    embed = [(n, p) for n, p in named if n.startswith("embed.")]
    other = [(n, p) for n, p in named if not n.startswith("embed.")]
    order = other + embed                       # same order as the optimizer groups in train.py
    lrs = []
    for gi, grp in enumerate(opt["param_groups"]):
        lrs += [(grp["lr"], grp.get("betas", (0.9, 0.999)), grp.get("eps", 1e-8))] * len(grp["params"])
    dsq, wsq = defaultdict(float), defaultdict(float)
    for idx, st in opt["state"].items():
        idx = int(idx)
        if idx >= len(order) or "exp_avg" not in st:
            continue
        n, p = order[idx]
        if n.endswith(".bias"):
            continue
        if st["exp_avg"].shape != p.shape:
            return {}                           # different parameter layout: skip quietly
        lr, (b1, b2), eps = lrs[idx]
        t = float(st["step"])
        m_hat = st["exp_avg"].float() / (1 - b1 ** t)
        v_hat = st["exp_avg_sq"].float() / (1 - b2 ** t)
        upd = lr * m_hat / (v_hat.sqrt() + eps)
        g = group_of(n)
        dsq[g] += float(upd.pow(2).sum())
        wsq[g] += float(p.detach().float().pow(2).sum())
    return {g: math.sqrt(dsq[g]) / (math.sqrt(wsq[g]) + 1e-12) for g in dsq}


def diagnose_checkpoint(ckpt, packs, n_batches=4, batch=8, device=None, importance=True, segment=16384):
    """Gradient flow, activation health and layer importance of a saved model."""
    from .audio import MelConfig, MelSketch
    from .model import FlowConfig, WaveFlow
    from tqdm.auto import tqdm
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model = WaveFlow(FlowConfig(**ck["flow_cfg"]), len(vocab))
    model.load_state_dict(ck["model"])
    model.to(device)
    mel = MelSketch(ck["mel_cfg"]).to(device)
    seg = segment // model.cfg.h * model.cfg.h
    data = _batches(packs, vocab, n_batches, batch, seg)

    # 1. gradients + activations (train mode, as during training)
    model.train()
    gsum, zsum = defaultdict(float), defaultdict(float)
    acts, flows = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    for x, y in tqdm(data, desc="gradients", leave=False):
        x, y = x.to(device), y.to(device)
        set_probes(model, True)
        model.zero_grad(set_to_none=True)
        model.nll(x, mel(x), y).backward()
        g, z = _group_norms(model, "grad")
        for k in g:
            gsum[k] += g[k] ** 2 / len(data)
            zsum[k] += z[k] / len(data)
        la, fl = _collect_probes(model)
        for k, d in la.items():
            for n, v in d.items():
                acts[k][n] += v / len(data)
        for k, d in fl.items():
            for n, v in d.items():
                flows[k][n] += v / len(data)
    set_probes(model, False)
    model.zero_grad(set_to_none=True)
    upd = _adam_update_ratio(model, ck)
    groups = {k: dict(g=math.sqrt(gsum[k]), zero=zsum[k], u=upd.get(k)) for k in gsum}

    # 2. layer importance (eval mode, fixed batches): val NLL increase when switched off
    imp = {}
    if importance:
        model.eval()

        @torch.no_grad()
        def nll():
            return sum(float(model.nll(x.to(device), mel(x.to(device)), y.to(device))) for x, y in data) / len(data)

        base = nll()
        L = model.cfg.n_layers
        jobs = [(k, None) for k in range(len(model.flows))] + \
               [(k, i) for k in range(len(model.flows)) for i in range(L)]
        for k, i in tqdm(jobs, desc="layer importance", leave=False):
            model.flows[k].ablate = "flow" if i is None else i
            imp[f"f{k}" if i is None else f"f{k}.L{i}"] = nll() - base
            model.flows[k].ablate = None
        imp["_base_nll"] = base
    return dict(step=ck.get("step"), ckpt=str(ckpt), n_flows=model.cfg.n_flows, n_layers=model.cfg.n_layers,
                has_adam=bool(upd), groups=groups,
                layers={k: dict(v) for k, v in acts.items()}, flows={k: dict(v) for k, v in flows.items()},
                importance=imp)


# ------------------------------------------------------------------ log.csv
def read_log(run_dir, with_grad=False):
    """(train, val[, grad]) as sorted (step, value) lists. A restarted run appends the same steps
    again; the last row for a step wins."""
    import csv
    p = Path(run_dir) / "log.csv"
    train, val, grad = {}, {}, {}
    if p.exists():
        for r in csv.DictReader(open(p)):
            try:
                s = int(r["step"])
                if r.get("train_nll"):
                    train[s] = float(r["train_nll"])
                if r.get("val_nll"):
                    val[s] = float(r["val_nll"])
                if r.get("grad_norm"):
                    grad[s] = float(r["grad_norm"])        # total norm BEFORE clipping
            except ValueError:
                continue
    out = [sorted(train.items()), sorted(val.items())]
    return (*out, sorted(grad.items())) if with_grad else tuple(out)


# ------------------------------------------------------------------ rules -> findings
def _f(status, area, check, value, meaning, action=""):
    return dict(status=status, area=area, check=check, value=value, meaning=meaning, action=action)


def _layer_keys(n_flows, n_layers):
    return [f"f{k}.L{i}" for k in range(n_flows) for i in range(n_layers)]


def analyze(snap, history, train, val, clip=1.0, ema_decay=0.999, grad_log=None):
    F = []
    nf, nl = snap["n_flows"], snap["n_layers"]
    lk = _layer_keys(nf, nl)
    groups = snap["groups"]
    step = snap.get("step") or 0

    # ---- gradient flow
    g = {k: groups[k]["g"] for k in lk if k in groups}
    if g:
        med = float(np.median(list(g.values())))
        dead = [k for k, v in g.items() if v < T["grad_dead"]]
        vanish = [k for k, v in g.items() if T["grad_dead"] <= v < T["grad_vanish"] * med]
        if dead and step < 10:
            F.append(_f(INFO, "Gradyan", "Sıfır gradyanlı katman", f"{len(dead)}/{len(g)}",
                        "Eğitimin başında beklenen: her flow'un son katmanı sıfırla başlatıldığı için gradyan "
                        "önce yalnız ona ulaşır, sonra geriye yayılır.", "Birkaç yüz adım sonra tekrar bak."))
        elif dead:
            F.append(_f(CRIT, "Gradyan", "Ölü gradyan (tam sıfır)", f"{len(dead)}/{len(g)}: {', '.join(dead[:8])}",
                        "Bu katmanlara hiç gradyan ulaşmıyor; ağırlıkları hiç değişmiyor.",
                        "Dondurulmuş mu kontrol et; değilse önündeki katmanların çıktısı sıfır olabilir (ölü kanallar)."))
        else:
            F.append(_f(OK, "Gradyan", "Ölü gradyan (tam sıfır)", f"0/{len(g)}", "Her katmana gradyan ulaşıyor."))
        if vanish:
            F.append(_f(WARN, "Gradyan", "Sönen gradyan", f"{len(vanish)} katman < medyanın %0,1'i: {', '.join(vanish[:8])}",
                        "Bu katmanlara ulaşan gradyan diğerlerinden yüzlerce kat küçük.",
                        "Güncelleme oranına bak: Adam küçük gradyanı telafi edebilir. O da düşükse lr ya da başlatma sorunu."))
        else:
            F.append(_f(OK, "Gradyan", "Sönen gradyan", "yok", "Katman gradyanları aynı büyüklük düzeninde."))
        per_flow = [np.sqrt(np.mean([g[f"f{k}.L{i}"] ** 2 for i in range(nl) if f"f{k}.L{i}" in g] or [0]))
                    for k in range(nf)]
        if per_flow[-1] > 0 and per_flow[0] > 0:
            r = per_flow[0] / per_flow[-1]
            st = CRIT if r < T["depth_crit"] else WARN if r < T["depth_warn"] or r > 1 / T["depth_crit"] else OK
            F.append(_f(st, "Gradyan", "Derinlik boyunca gradyan (ilk flow / son flow)", f"{r:.3g}",
                        "1'e yakınsa gradyan girişe kadar sönmeden ulaşıyor. < 0,1: girişe doğru sönüyor; "
                        "> 100: girişe doğru büyüyor (patlama).",
                        "" if st == OK else "Flow sayısını azalt ya da lr/warmup'ı gözden geçir."))
    # update ratio (training history preferred, else Adam state)
    upd = {}
    if history:
        recent = history[-min(5, len(history)):]
        for k in lk:
            vals = [h["groups"][k]["u"] for h in recent if k in h["groups"]]
            if vals:
                upd[k] = float(np.median(vals))
        src = "eğitim kaydı"
    else:
        upd = {k: groups[k]["u"] for k in lk if k in groups and groups[k].get("u") is not None}
        src = "Adam durumu (last.pt)"
    if upd:
        low = [k for k, v in upd.items() if v < T["upd_low"]]
        high = [k for k, v in upd.items() if v > T["upd_high"]]
        med = float(np.median(list(upd.values())))
        if high:
            F.append(_f(CRIT, "Gradyan", "Güncelleme oranı çok büyük", f"{len(high)} katman > 0,1 ({src})",
                        "Her adımda ağırlıkların %10'undan fazlası değişiyor: eğitim kararsız olabilir.", "lr'yi düşür."))
        if low:
            F.append(_f(WARN, "Gradyan", "Öğrenmeyen katman (güncelleme < 1e-5)", f"{len(low)}: {', '.join(low[:8])}",
                        "Gradyan olsa bile bu katmanların ağırlıkları neredeyse hiç değişmiyor.",
                        "Eğitim sonundaysa (lr düşük) normal; ortasındaysa lr ya da gradyan akışı sorunlu."))
        if not low and not high:
            F.append(_f(OK, "Gradyan", "Güncelleme oranı ||Δw|| / ||w||", f"medyan {med:.1e} ({src})",
                        "Sağlıklı aralık kabaca 1e-4 – 1e-2: ağırlıklar ne donuk ne de savruluyor."))
    elif not history:
        F.append(_f(INFO, "Gradyan", "Güncelleme oranı", "ölçülemedi",
                    "Checkpoint'te optimizer durumu yok (best.pt). last.pt ile ya da eğitim kaydıyla ölçülür."))
    if not history and grad_log:
        gn = np.array([v for _, v in grad_log])
        frac = float(np.mean(gn > clip))
        st = WARN if frac > T["clip_warn"] else OK
        F.append(_f(st, "Gradyan", "Gradyan kırpma sıklığı (log.csv)",
                    f"kayıtların %{100 * frac:.0f}'i > {clip}; kırpma öncesi norm medyanı {np.median(gn):.1f}, "
                    f"en büyük {gn.max():.0f}",
                    f"Toplam gradyan normu {clip} sınırını aştığında kırpılıyor. Neredeyse her adımda kırpılıyorsa "
                    "adım boyunu lr değil kırpma belirliyor ve büyük gradyanlı adımlar orantısız küçülüyor.",
                    "" if st == OK else "grad_clip'i tipik normun biraz üstüne çek (ör. medyanın 2-3 katı) ya da "
                    "lr'yi düşür; patlama yoksa kırpma yalnız ani sıçramaları kesmeli."))
    if history:
        clip_frac = float(np.mean([h["clip_frac"] for h in history[-10:]]))
        nonfinite = sum(h["nonfinite"] for h in history)
        st = WARN if clip_frac > T["clip_warn"] else OK
        F.append(_f(st, "Gradyan", "Gradyan kırpma sıklığı", f"adımların %{100 * clip_frac:.0f}'i",
                    f"Toplam gradyan normu {clip} sınırını aştığında kırpılıyor. Çoğu adımda kırpılıyorsa "
                    "gerçek adım boyunu lr değil kırpma belirliyor.",
                    "" if st == OK else "lr'yi düşür ya da grad_clip'i artır."))
        if nonfinite:
            F.append(_f(CRIT, "Gradyan", "NaN / sonsuz gradyan", f"{nonfinite} adım",
                        "Sayısal patlama: bu adımlar atlandı.", "lr'yi düşür, fp32 dene, veride bozuk klip ara."))

    # ---- neurons
    layers = snap["layers"]
    if history and history[-1].get("layers"):
        layers = history[-1]["layers"]
    if layers:
        def worst(key):
            items = sorted(((layers[k].get(key, 0.0), k) for k in layers), reverse=True)
            return items, float(np.mean([v for v, _ in items]))
        items, mean = worst("dead_ch")
        bad = [f"{k} %{100 * v:.0f}" for v, k in items if v > T["dead_warn"]]
        st = CRIT if items[0][0] > T["dead_crit"] else WARN if bad else OK
        F.append(_f(st, "Nöron", "Ölü kanal (her girdide ~0 çıktı)",
                    f"ortalama %{100 * mean:.1f}" + (f"; {', '.join(bad[:6])}" if bad else ""),
                    "Ölü kanal hiçbir şey taşımaz ve gradyan da almaz; kapasite boşa gider.",
                    "" if st == OK else "lr'yi düşür; dropout'u kapat; ilgili katmanların kapı girişlerine bak."))
        items, mean = worst("tanh_sat")
        bad = [f"{k} %{100 * v:.0f}" for v, k in items if v > T["sat_warn"]]
        F.append(_f(WARN if bad else OK, "Nöron", "Doymuş tanh (|giriş| > 2,5)",
                    f"ortalama %{100 * mean:.1f}" + (f"; {', '.join(bad[:6])}" if bad else ""),
                    "Doymuş tanh'ın türevi ~0'dır: gradyan oradan geçemez (sönen gradyanın klasik sebebi).",
                    "" if not bad else "lr'yi düşür; koşul (taslak) girişinin ölçeğini kontrol et."))
        items, mean = worst("gate_closed")
        bad = [f"{k} %{100 * v:.0f}" for v, k in items if v > T["gate_warn"]]
        F.append(_f(WARN if bad else OK, "Nöron", "Kapalı sigmoid kapısı (σ < 0,02)",
                    f"ortalama %{100 * mean:.1f}" + (f"; {', '.join(bad[:6])}" if bad else ""),
                    "Kapı kapalıyken birim çıktı vermez; çoğu zaman kapalıysa birim fiilen ölüdür."))
        cr = [v["cond_ratio"] for v in layers.values() if "cond_ratio" in v]
        if cr:
            med = float(np.median(cr))
            st = CRIT if med < 0.05 else WARN if med < 0.15 else OK
            F.append(_f(st, "Koşul", "Taslak + etiket sinyalinin gücü (ses yoluna oranı)",
                        f"medyan {med:.2f} (en düşük {min(cr):.2f}, en yüksek {max(cr):.2f})",
                        "Kapı girişinde koşuldan gelen sinyalin ses yolundan gelene oranı. Çok küçükse model taslağı "
                        "ve dolayısıyla parametreleri neredeyse duymaz (base_v1'in ilk katmanlarında ~0,02–0,1 idi).",
                        "" if st == OK else "cond_norm / actnorm açık mı kontrol et; koşulu FiLM ile güçlendirmeyi dene."))
    fl = snap["flows"]
    if history and history[-1].get("flows"):
        fl = history[-1]["flows"]
    if fl:
        rd = {k: v["relu_dead"] for k, v in fl.items()}
        bad = [f"{k} %{100 * v:.0f}" for k, v in rd.items() if v > T["relu_warn"]]
        F.append(_f(WARN if bad else OK, "Nöron", "Ölü ReLU (son 1×1 konvolüsyondan önce)",
                    f"ortalama %{100 * np.mean(list(rd.values())):.1f}" + (f"; {', '.join(bad)}" if bad else ""),
                    "Skip toplamı hep negatif olan kanallar ReLU'da sıfırlanır; son katman onları göremez."))

    # ---- unused layers
    imp = snap.get("importance") or {}
    li = {k: imp[k] for k in lk if k in imp}
    if li:
        pos = [v for v in li.values() if v > 0]
        med = float(np.median(pos)) if pos else 0.0
        unused = [k for k, v in li.items() if v < T["unused_rel"] * med]
        harmful = [k for k, v in li.items() if v < 0]
        n_unused = len(unused)
        st = WARN if n_unused > len(li) * 0.25 else INFO if n_unused else OK
        F.append(_f(st, "Mimari", "Katkısı ihmal edilebilir katman",
                    f"{n_unused}/{len(li)}" + (f": {', '.join(unused[:10])}" if unused else ""),
                    "Katman kapatılınca val NLL medyan katmanın %1'inden az artıyor: model onu kullanmıyor.",
                    "" if not n_unused else "Çoksa: n_layers'ı azalt (aynı compute ile daha uzun eğit) ya da eğitimin "
                    "ilerlemesini bekle (erken checkpoint'lerde normal)."))
        if harmful:
            F.append(_f(WARN, "Mimari", "Kapatınca iyileşen katman", ", ".join(harmful[:10]),
                        "Bu katman olmadan val NLL daha iyi: katman zarar veriyor ya da aşırı öğrenmiş.",
                        "Birkaç checkpoint'te tekrarlıyorsa katman/flow sayısını azaltmayı dene."))
        fi = {f"f{k}": imp.get(f"f{k}") for k in range(nf) if f"f{k}" in imp}
        if fi:
            weakest = min(fi, key=fi.get)
            F.append(_f(INFO, "Mimari", "En az katkılı flow", f"{weakest}: ΔNLL {fi[weakest]:.4f}",
                        "Bir flow'u tamamen kapatmanın val NLL'e etkisi; diğer flow'larla kıyasla (grafik)."))

    # ---- fit
    F += _fit_findings(train, val)
    settle = int(3 / max(1e-9, 1 - ema_decay))
    if val and val[-1][0] < settle:
        F.append(_f(INFO, "Uyum", "EMA henüz oturmadı", f"son val adımı {val[-1][0]} < {settle}",
                    f"Val NLL, ağırlıkların yavaş hareketli ortalamasıyla (EMA {ema_decay}) ölçülüyor; ilk ~{settle} "
                    "adımda EMA başlangıç ağırlıklarına yakın kalır. Bu aralıkta val'in geride kalması ve val − train "
                    "farkının büyümesi normaldir; overfit/underfit kararı güvenilir değil."))
    order = {CRIT: 0, WARN: 1, INFO: 2, OK: 3}
    return sorted(F, key=lambda f: order[f["status"]])


def _fit_findings(train, val):
    F = []
    if len(val) < 3:
        return [_f(INFO, "Uyum", "Overfit / underfit", f"{len(val)} doğrulama noktası",
                   "Karar için en az 3 doğrulama noktası gerekiyor (eval_every adımda bir).")]
    vs = np.array([v for _, v in val])
    steps = np.array([s for s, _ in val])
    best_i = int(np.argmin(vs))
    tol = 0.002 + 0.002 * abs(vs[best_i])
    tr = np.array([v for _, v in train]) if train else np.array([])
    def train_at(s):
        near = [v for st, v in train if s - 1000 < st <= s]
        return float(np.mean(near)) if near else float("nan")
    gap_first, gap_last = vs[0] - train_at(steps[0]), vs[-1] - train_at(steps[-1])
    train_falling = len(tr) >= 10 and np.mean(tr[-len(tr) // 5:]) < np.mean(tr[-2 * len(tr) // 5: -len(tr) // 5]) - tol
    if best_i < len(vs) - 2 and vs[-1] > vs[best_i] + tol:
        st = CRIT if vs[-1] > vs[best_i] + 5 * tol else WARN
        F.append(_f(st, "Uyum", "Aşırı öğrenme (overfit)",
                    f"en iyi val {vs[best_i]:.4f} @ {steps[best_i]}, son {vs[-1]:.4f} @ {steps[-1]}",
                    "Val NLL en iyi noktasından sonra yükseliyor" + (" ve train NLL düşmeye devam ediyor" if train_falling else "")
                    + ": model eğitim verisini ezberlemeye başladı.",
                    "best.pt'yi kullan; daha çok veri, dropout (0,05–0,1) ya da augmentation; daha az adım."))
    else:
        slope = (vs[-1] - vs[-3]) / max(steps[-1] - steps[-3], 1) * 1000
        if slope < -tol:
            F.append(_f(WARN, "Uyum", "Az öğrenme / henüz yakınsamadı (underfit)",
                        f"val son 3 noktada 1000 adımda {slope:+.4f} değişiyor",
                        "Val NLL hâlâ belirgin şekilde düşüyor: model öğrenmeye devam ediyordu, eğitim erken kesildi.",
                        "Aynı RUN adıyla devam ettir (STEPS / MAX_HOURS artır). Dropout'u açma."))
        else:
            F.append(_f(OK, "Uyum", "Yakınsama", f"val son 3 noktada 1000 adımda {slope:+.4f}",
                        "Val NLL düzleşti ve yükselmiyor: ne aşırı ne az öğrenme işareti var.",
                        "Kalite hâlâ yetersizse sorun kapasite ya da veri: model boyutu / veri miktarı."))
    if math.isfinite(gap_first) and math.isfinite(gap_last):
        F.append(_f(INFO, "Uyum", "Val − train farkı (ilk → son)", f"{gap_first:+.4f} → {gap_last:+.4f}",
                    "Train NLL augmentation ve etiket dropout'lu ham modelden, val NLL EMA modelden ölçülüyor; "
                    "bu yüzden mutlak fark değil, zamanla büyümesi önemli."))
    return F


# ------------------------------------------------------------------ report
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
_S1, _S2 = "#2a78d6", "#eb6834"                      # categorical slots 1, 2
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_STATUS = {OK: "#0ca30c", WARN: "#fab219", CRIT: "#d03b3b", INFO: "#8a8984"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE, "axes.edgecolor": _GRID,
                         "axes.labelcolor": _INK2, "xtick.color": _INK2, "ytick.color": _INK2,
                         "text.color": _INK, "axes.grid": True, "grid.color": _GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})
    return plt


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _heatmap(plt, M, rows, cols, title, fmt, log=False, vmin=None, vmax=None, note=""):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    cmap = LinearSegmentedColormap.from_list("seq", _SEQ)
    data = np.array(M, dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 0.9 + 0.42 * len(rows)))
    finite = data[np.isfinite(data)]
    if log:
        pos = finite[finite > 0]
        lo = vmin or (pos.min() if len(pos) else 1e-8)
        hi = vmax or (pos.max() if len(pos) else 1.0)
        norm = LogNorm(lo, max(hi, lo * 10))
        shown = np.where(data > 0, data, lo)
    else:
        norm = Normalize(vmin if vmin is not None else (finite.min() if len(finite) else 0),
                         vmax if vmax is not None else (finite.max() if len(finite) else 1))
        shown = data
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(data):
        if np.isfinite(v):
            dark = norm(shown[r, c]) > 0.55
            ax.text(c, r, fmt(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else _INK)
    ax.set_title(title, loc="left")
    if note:
        ax.set_xlabel(note, fontsize=8.5)
    return _png(fig)


def _figures(snap, history, train, val):
    plt = _plt()
    figs = []
    nf, nl = snap["n_flows"], snap["n_layers"]
    rows = [f"flow {k}" for k in range(nf)]
    # loss
    if train or val:
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        if train:
            s, v = zip(*train)
            v = np.array(v, dtype=float)
            v = np.array([v[max(0, i - 9): i + 1].mean() for i in range(len(v))])     # trailing mean, no gap at start
            ax.plot(s, v, color=_S1, lw=2, label="train NLL (son 10 kaydın ortalaması)")
        if val:
            s, v = zip(*val)
            ax.plot(s, v, color=_S2, lw=2, marker="o", ms=5, label="val NLL (EMA)")
            b = int(np.argmin(v))
            ax.annotate(f"en iyi {v[b]:.4f}", (s[b], v[b]), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=8.5, color=_INK2)
        ax.set_xlabel("adım"); ax.set_ylabel("nats / örnek")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.set_title("Kayıp eğrileri", loc="left")
        figs.append(("Kayıp eğrileri", _png(fig),
                     "İkisi birlikte düşüyorsa öğrenme sürüyor; val yükselirken train düşüyorsa aşırı öğrenme."))
    # gradient heatmap
    extra = ["start", "cond", "label", "end"]
    cols = [f"L{i}" for i in range(nl)] + extra
    G = [[snap["groups"].get(f"f{k}.{c}", {}).get("g", np.nan) for c in cols] for k in range(nf)]
    figs.append(("Gradyan akışı", _heatmap(plt, G, rows, cols, "Gradyan normu (log ölçek)",
                                            lambda v: f"{v:.0e}", log=True,
                                            note="L0–L7: gated katmanlar · cond: taslak girişi · label: kategori · end: çıkış"),
                 "Her hücre bir katmanın gradyan büyüklüğü. Bir satır ya da sütun diğerlerinden çok açıksa gradyan "
                 "oraya sönerek ulaşıyor; boş (beyaz) hücre sıfır gradyan."))
    U = [[(snap["groups"].get(f"f{k}.{c}", {}) or {}).get("u") for c in cols] for k in range(nf)]
    if history:
        recent = history[-1]["groups"]
        U = [[recent.get(f"f{k}.{c}", {}).get("u") for c in cols] for k in range(nf)]
    U = [[np.nan if v is None else v for v in row] for row in U]
    if np.isfinite(np.array(U, dtype=float)).any():
        figs.append(("Güncelleme oranı", _heatmap(plt, U, rows, cols, "Güncelleme oranı ||Δw|| / ||w|| (log ölçek)",
                                                   lambda v: f"{v:.0e}", log=True, vmin=1e-6, vmax=1e-1),
                     "Ağırlıkların bir adımda ne kadar değiştiği. Sağlıklı aralık ~1e-4 – 1e-2; 1e-5 altı donuk."))
    # history lines
    if len(history) >= 2:
        steps = [h["step"] for h in history]
        lk = _layer_keys(nf, nl)
        def band(key):
            arr = np.array([[h["groups"].get(k, {}).get(key, np.nan) for k in lk] for h in history], dtype=float)
            arr[arr <= 0] = np.nan          # step 1: zero by design (zero-initialised output layers); log axis
            return np.nanmin(arr, 1), np.nanmedian(arr, 1), np.nanmax(arr, 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, key, title in [(axes[0], "g", "Gradyan normu"), (axes[1], "u", "Güncelleme oranı")]:
            lo, med, hi = band(key)
            ax.fill_between(steps, lo, hi, color=_S1, alpha=0.15, lw=0, label="en küçük – en büyük katman")
            ax.plot(steps, med, color=_S1, lw=2, label="katman medyanı")
            if key == "u":
                ax.axhspan(1e-4, 1e-2, color=_STATUS[OK], alpha=0.07, lw=0, label="sağlıklı aralık")
            ax.set_yscale("log"); ax.set_xlabel("adım"); ax.set_title(title, loc="left")
        for ax in axes:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
        figs.append(("Zaman içinde gradyanlar", _png(fig),
                     "Band çok genişliyorsa bazı katmanlar diğerlerinden kopuyor; medyan sıfıra yaklaşıyorsa gradyan ölüyor."))
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        dead = [np.mean([v.get("dead_ch", 0) for v in h["layers"].values()]) * 100 if h.get("layers") else np.nan
                for h in history]
        sat = [np.mean([v.get("tanh_sat", 0) for v in h["layers"].values()]) * 100 if h.get("layers") else np.nan
               for h in history]
        ax.plot(steps, dead, color=_S1, lw=2, label="ölü kanal %")
        ax.plot(steps, sat, color=_S2, lw=2, label="doymuş tanh %")
        ax.set_xlabel("adım"); ax.set_ylabel("%")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.set_title("Nöron sağlığı (tüm katmanların ortalaması)", loc="left")
        figs.append(("Zaman içinde nöronlar", _png(fig), "Artan bir eğri, ağın bir kısmının zamanla kullanılmaz hale geldiğini gösterir."))
    # neurons heatmap
    layers = history[-1]["layers"] if history and history[-1].get("layers") else snap["layers"]
    if layers:
        lc = [f"L{i}" for i in range(nl)]
        D = [[100 * layers.get(f"f{k}.L{i}", {}).get("dead_ch", np.nan) for i in range(nl)] for k in range(nf)]
        S = [[100 * layers.get(f"f{k}.L{i}", {}).get("tanh_sat", np.nan) for i in range(nl)] for k in range(nf)]
        figs.append(("Ölü kanallar", _heatmap(plt, D, rows, lc, "Ölü kanal yüzdesi", lambda v: f"{v:.0f}",
                                              vmin=0, vmax=max(30, np.nanmax(D) if np.isfinite(D).any() else 30)),
                     "Her girdide ~0 çıktı veren kanalların payı. %10 üstü uyarı, %30 üstü kritik."))
        figs.append(("Doymuş tanh", _heatmap(plt, S, rows, lc, "Doymuş tanh yüzdesi", lambda v: f"{v:.0f}",
                                             vmin=0, vmax=max(30, np.nanmax(S) if np.isfinite(S).any() else 30)),
                     "Türevi ~0 olan bölgeye düşen tanh girişlerinin payı. %30 üstü uyarı."))
    # importance
    imp = snap.get("importance") or {}
    if any(k.startswith("f") and ".L" in k for k in imp):
        I = [[imp.get(f"f{k}.L{i}", np.nan) for i in range(nl)] for k in range(nf)]
        arr = np.array(I, dtype=float)
        figs.append(("Katman önemi", _heatmap(plt, np.where(arr > 0, arr, np.nan) if (arr > 0).any() else arr, rows,
                                              [f"L{i}" for i in range(nl)],
                                              "Katman kapatılınca val NLL artışı (log ölçek)", lambda v: f"{v:.0e}", log=True),
                     "Koyu: model bu katmana çok dayanıyor. Açık: katkısı az. Boş: kapatınca NLL düşüyor (zararlı)."))
        fi = [imp.get(f"f{k}", np.nan) for k in range(nf)]
        fig, ax = plt.subplots(figsize=(8.5, 2.8))
        ax.bar(range(nf), fi, color=_S1, width=0.6)
        ax.set_xticks(range(nf), rows); ax.set_ylabel("Δ val NLL")
        ax.set_title("Flow kapatılınca val NLL artışı", loc="left")
        for k, v in enumerate(fi):
            if np.isfinite(v):
                ax.text(k, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=_INK2)
        figs.append(("Flow önemi", _png(fig), "Her flow tamamen kapatıldığında (birim dönüşüm) kaybın ne kadar arttığı."))
    return figs


def _html(title, findings, figs, meta):
    import html as H
    rows = "".join(
        f"<tr><td><span class='st' style='--c:{_STATUS[f['status']]}'>{_ICON[f['status']]} {f['status']}</span></td>"
        f"<td>{H.escape(f['area'])}</td><td><b>{H.escape(f['check'])}</b><div class='m'>{H.escape(f['meaning'])}</div>"
        + (f"<div class='a'>→ {H.escape(f['action'])}</div>" if f["action"] else "")
        + f"</td><td class='v'>{H.escape(str(f['value']))}</td></tr>" for f in findings)
    counts = {s: sum(f["status"] == s for f in findings) for s in (CRIT, WARN, OK, INFO)}
    tiles = "".join(f"<div class='tile'><div class='n' style='color:{_STATUS[s]}'>{n}</div><div>{s}</div></div>"
                    for s, n in counts.items())
    sections = "".join(f"<section><h2>{H.escape(t)}</h2><p class='m'>{H.escape(cap)}</p>"
                       f"<img alt='{H.escape(t)}' src='data:image/png;base64,{b64}'></section>" for t, b64, cap in figs)
    thr = "".join(f"<li><code>{k}</code> = {v}</li>" for k, v in T.items())
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{H.escape(title)}</title>
<style>
:root{{color-scheme:light;--surface:{_SURFACE};--ink:{_INK};--ink2:{_INK2};--grid:{_GRID}}}
body{{margin:0;background:var(--surface);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 4px}}
.m{{color:var(--ink2);font-size:13px}} .a{{font-size:13px;margin-top:2px}}
.tiles{{display:flex;gap:12px;margin:16px 0;flex-wrap:wrap}}
.tile{{border:1px solid var(--grid);border-radius:8px;padding:8px 16px;min-width:80px;text-align:center;color:var(--ink2)}}
.tile .n{{font-size:26px;font-weight:700}}
table{{width:100%;border-collapse:collapse}} td{{border-top:1px solid var(--grid);padding:8px 6px;vertical-align:top}}
td.v{{font-variant-numeric:tabular-nums;white-space:normal;max-width:260px;color:var(--ink2);font-size:13px}}
.st{{display:inline-block;white-space:nowrap;font-weight:600;font-size:12px;padding:2px 8px;border-radius:10px;
     border:1.5px solid var(--c);color:var(--ink)}}
img{{max-width:100%;height:auto;border:1px solid var(--grid);border-radius:6px}}
.wrap{{overflow-x:auto}} details{{margin-top:24px;color:var(--ink2);font-size:13px}}
</style></head><body><main>
<h1>{H.escape(title)}</h1><div class="m">{H.escape(meta)}</div>
<div class="tiles">{tiles}</div>
<div class="wrap"><table><tbody>{rows}</tbody></table></div>
{sections}
<details><summary>Eşikler</summary><ul>{thr}</ul></details>
</main></body></html>"""


def _text_table(findings):
    lines = []
    for f in findings:
        lines.append(f"[{f['status']:6s}] {f['area']:8s} {f['check']}: {f['value']}")
        if f["status"] in (WARN, CRIT) and f["action"]:
            lines.append(f"{'':17s}→ {f['action']}")
    return "\n".join(lines)


def report(run_dir, packs=None, ckpt="last.pt", n_batches=4, device=None, importance=True, batch=8, segment=16384):
    """Build <run>/health/report.html (+ findings.json). Returns (findings, html_path)."""
    run = Path(run_dir)
    ck = run / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
    if not ck.exists():
        ck = run / "best.pt"
    history = read_history(run)
    train, val, grad_log = read_log(run, with_grad=True)
    if packs is None:
        cfg = torch.load(ck, map_location="cpu", weights_only=False).get("train_cfg", {})
        packs = cfg.get("packs", [])
    snap = diagnose_checkpoint(ck, packs, n_batches=n_batches, batch=batch, device=device, importance=importance,
                               segment=segment)
    tcfg = torch.load(ck, map_location="cpu", weights_only=False).get("train_cfg") or {}
    findings = analyze(snap, history, train, val, clip=tcfg.get("grad_clip", 1.0),
                       ema_decay=tcfg.get("ema_decay", 0.999), grad_log=grad_log)
    figs = _figures(snap, history, train, val)
    out = run / "health"
    out.mkdir(exist_ok=True)
    meta = (f"{run.name} · checkpoint {ck.name} (adım {snap['step']}) · "
            f"{'eğitim kaydı: ' + str(len(history)) + ' ölçüm' if history else 'eğitim kaydı yok (yalnız checkpoint analizi)'}")
    path = out / "report.html"
    path.write_text(_html(f"Eğitim sağlık raporu: {run.name}", findings, figs, meta))
    (out / "findings.json").write_text(json.dumps(dict(findings=findings, snapshot=snap), indent=1, default=float))
    print(_text_table(findings))
    print("\nrapor:", path)
    return findings, path
