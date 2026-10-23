"""Sounds from scratch: vector -> mel generator -> (controls) -> WaveFlow vocoder -> audio. No reference sound.

    rows = from_scratch("runs/melgen_v1/mel_gen.pt", "runs/vocoder_v1/best.pt", jobs, "out/", packs=[pack])
    page(rows, "out/index.html", "melgen_v1 · sıfırdan")

A job is a dict:  name, and ONE of  label="gunshot" | mix={"gunshot": .7, "thunder": .3} | vec=<tensor [vec_dim]>,
plus optional seed, controls (Controls), guidance, frames.

Novelty: each generated mel is compared with real mels from the packs (nearest neighbour over time offsets, mean |dB|
over loud cells). The same distance between two different real clips is the reference: a generated mel about as
far from its nearest real clip as real clips are from each other is new, not a copy.
"""
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import torch

from .controls import Controls
from .sketch_model import SketchDiffusion


def _clip_mels(packs, mel_cfg, split, per_label, max_s, seed, device):
    """{label: [(clip_id, mel [n_mels, F] on device)]} from a pack split."""
    from .audio import MelConfig, MelSketch
    from .data import read_index
    mel = MelSketch(MelConfig(**mel_cfg)).to(device)
    rng = np.random.default_rng(seed)
    out = {}
    for p in packs:
        audio = np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r")
        rows = [r for r in read_index(p) if r["split"] == split]
        rng.shuffle(rows)
        for r in rows:
            lst = out.setdefault(r["label"], [])
            if per_label and len(lst) >= per_label:
                continue
            n = min(int(r["length"]), int(max_s * mel_cfg["sample_rate"]))
            a = torch.from_numpy(np.asarray(audio[int(r["offset"]): int(r["offset"]) + n], np.float32) / 32768)
            lst.append((r["id"], mel(a[None].to(device))[0]))
    return out


class NoveltyIndex:
    """Nearest training clip for a generated mel: every window of up to `per_label` training
    clips per label (every frame offset), ALL labels (a copy could come from any category). Distance = mean |dB| over cells where
    either side is louder than -60 dB. Reference: the same distance for real recordings the generator never saw
    (val split) -> "a genuinely new recording is about this far from its nearest training clip"."""

    def __init__(self, packs, mel_cfg, frames, per_label=150, max_s=6.0, device=None, stride=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frames, self.stride, self.mel_cfg = frames, stride, mel_cfg
        train = _clip_mels(packs, mel_cfg, "train", per_label, max_s, 0, self.device)
        self.val = _clip_mels(packs, mel_cfg, "val", 8, max_s, 1, self.device)
        # one long tensor: clips back to back, separated by `frames` silent frames (a window that straddles two
        # clips is far from everything), plus per-frame owner -> one batched search instead of a loop per clip
        parts, owner, self.owners = [], [], []
        for label, clips in train.items():
            for cid, m in clips:
                k = len(self.owners)
                self.owners.append((label, cid))
                parts += [m, torch.zeros(m.shape[0], frames, device=self.device)]
                owner += [k] * m.shape[1] + [-1] * frames
        # dB, fp16 on GPU (60 labels x 150 clips x ~8 s of frames ~ 2 GB)
        self.bank = (torch.cat(parts, 1) * 100 - 100).to(torch.float16 if self.device != "cpu" else torch.float32) if parts else None
        self.owner = torch.tensor(owner, device=self.device)
        self.n = len(self.owners)

    @torch.no_grad()
    def nearest(self, m, skip_id=None, block=None):
        """(distance dB, label, clip id) of the closest training window, every frame offset."""
        if self.bank is None:
            return (None, None, None)
        a = torch.as_tensor(m, dtype=torch.float32, device=self.device) * 100 - 100        # [C, T]
        T = a.shape[1]
        b = self.bank
        if b.shape[1] < T:
            b = torch.nn.functional.pad(b, (0, T - b.shape[1]), value=-100.0)
        w_all = b.unfold(1, T, self.stride)                                              # [C, n_win, T] (view)
        start_owner = self.owner[:: self.stride][: w_all.shape[1]]
        skip = None
        if skip_id is not None:
            skip = [k for k, (_, cid) in enumerate(self.owners) if cid == skip_id]
        best_d, best_i = float("inf"), -1
        aa, la = a[:, None], a[:, None] > -60
        block = block or (8192 if self.device != "cpu" else 2048)
        for s0 in range(0, w_all.shape[1], block):
            w = w_all[:, s0: s0 + block].float()
            mk = la | (w > -60)
            d = ((aa - w).abs() * mk).sum((0, 2)) / mk.sum((0, 2)).clamp_min(1)
            own = start_owner[s0: s0 + block]
            d = torch.where(own < 0, torch.full_like(d, float("inf")), d)               # starts in a gap
            if skip:
                d = torch.where(torch.isin(own, torch.tensor(skip, device=d.device)), torch.full_like(d, float("inf")), d)
            i = int(d.argmin())
            if float(d[i]) < best_d:
                best_d, best_i = float(d[i]), s0 + i
        label, cid = self.owners[int(start_owner[best_i])]
        return (best_d, label, cid)

    def baseline(self, n=24, seed=1):
        """Median nearest-training-clip distance of unseen real recordings (val split), same crop length."""
        rng = np.random.default_rng(seed)
        pool = [(l, cid, m) for l, v in self.val.items() for cid, m in v if m.shape[1] >= self.frames]
        if not pool:
            return None
        ds = []
        for k in rng.permutation(len(pool))[:n]:
            _, cid, m = pool[int(k)]
            pk = int(m.mean(0).argmax())
            st = int(np.clip(pk - self.frames // 3, 0, m.shape[1] - self.frames))
            ds.append(self.nearest(m[:, st: st + self.frames].cpu().numpy())[0])
        return float(np.median(ds))


def from_scratch(mel_gen, vocoder, jobs, out_dir, packs=None, steps=50, guidance=2.0, temperature=1.0,
                 device=None, per_label_ref=150):
    from .generate import Generator
    from .audio import save_audio
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mg, ck = SketchDiffusion.load(mel_gen, device)
    gen = Generator(vocoder, device=device)
    if ck.get("mel_cfg") and ck["mel_cfg"] != gen.mel_cfg:
        raise ValueError(f"mel contract mismatch: generator {ck['mel_cfg']} vs vocoder {gen.mel_cfg}")
    if not gen.vocoder:
        print("note: WaveFlow checkpoint is category-conditioned, not a universal vocoder")
    index = NoveltyIndex(packs, gen.mel_cfg, mg.cfg.frames, per_label=per_label_ref, device=device) if packs else None
    real_to_real = index.baseline() if index else None
    if index:
        print(f"novelty index: {index.n} training clips; unseen real recordings are {real_to_real:.1f} dB "
              f"from their nearest training clip (median)")
    rows = []
    for j in jobs:
        seed = j.get("seed", 0)
        if "vec" in j:
            vec = torch.as_tensor(j["vec"], dtype=torch.float32)
        else:
            vec = mg.vector(label=j.get("label"), mix=j.get("mix"))
        g = torch.Generator(device=device).manual_seed(seed)
        m = mg.sample(vec, n=1, steps=steps, guidance=j.get("guidance", guidance), frames=j.get("frames"),
                      generator=g)[0].cpu().numpy()
        kind_label = j.get("label") or (max(j["mix"], key=j["mix"].get) if j.get("mix") else "event")
        r = gen.generate(kind_label, j.get("controls") or Controls(), seed=seed, temperature=temperature, mel=m)
        a = r["audio"][0]
        f = out / f"{j['name']}.wav"
        save_audio(f, a, gen.sr)
        nov, nl, nid = index.nearest(m) if index else (None, None, None)
        rows.append(dict(name=j["name"], wav=str(f), mel=m, edited=r["sketch"], seconds=len(a) / gen.sr,
                         spec={k: v for k, v in j.items() if k in ("label", "mix", "seed")},
                         nearest_db=nov, nearest_label=nl, nearest_id=nid, real_to_real_db=real_to_real))
        print(f"{j['name']:28s} {len(a) / gen.sr:.2f}s  nearest training clip "
              f"{'-' if nov is None else f'{nov:.1f} dB ({nl})'}  | unseen real: "
              f"{'-' if real_to_real is None else f'{real_to_real:.1f} dB'}")
    meta = [{k: v for k, v in r.items() if k not in ("mel", "edited")} for r in rows]
    meta.append(dict(real_to_real_db=real_to_real, index_clips=index.n if index else 0))
    (out / "scratch.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return rows


def _png(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.2, 1.3), dpi=100)
    ax.imshow(m, origin="lower", aspect="auto", cmap="magma", vmin=0.2, vmax=1.1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ogg(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="OGG", subtype="VORBIS")
    return "data:audio/ogg;base64," + base64.b64encode(buf.getvalue()).decode()


def page(rows, dest, title, zip_path=None):
    """Plain showcase page: mel + player per sound (no reference audio anywhere). Optional zip of the wavs."""
    trs = []
    for r in rows:
        sp = r["spec"]
        what = sp.get("label") or " + ".join(f"{k} {v:.0%}" for k, v in (sp.get("mix") or {}).items()) or "vektör"
        trs.append(f"<tr><td class='lab'>{html.escape(r['name'])}<span class='k'>{html.escape(what)} · tohum "
                   f"{sp.get('seed', 0)}</span>"
                   + (f"<span class='k'>en yakın eğitim kaydı: {r['nearest_db']:.1f} dB ({html.escape(str(r['nearest_label']))})</span>"
                      if r.get("nearest_db") is not None else "")
                   + (f"<span class='k'>CLAP: {html.escape(r['clap'])}</span>" if r.get("clap") else "")
                   + f"</td><td><img alt='' src='{_png(r['mel'])}'></td>"
                   f"<td><audio controls preload='none' src='{_ogg(r['wav'])}'></audio></td></tr>")
    ref = next((r["real_to_real_db"] for r in rows if r.get("real_to_real_db") is not None), None)
    intro = (f"<p class='k'>Hiçbir ses referans kullanmadı. Kopya kontrolü: eğitimde hiç görülmemiş gerçek kayıtlar "
             f"en yakın eğitim kaydına ortalama {ref:.1f} dB uzaklıkta; üretilenler bu değere yakın ya da büyükse yeni, "
             f"çok küçükse kopyadır.</p>" if ref is not None else "")
    css = """:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e7e6e2}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 12px}
table{border-collapse:collapse;width:100%}td{border-top:1px solid var(--rule);padding:8px 6px;vertical-align:middle}
.lab{font-weight:600}.k{display:block;font-size:12px;font-weight:400;color:var(--ink2)}img{width:240px;height:auto;border-radius:4px;display:block}
audio{height:34px;max-width:260px}.wrap{overflow-x:auto}"""
    doc = (f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style>"
           f"</head><body><main><h1>{html.escape(title)}</h1>{intro}<div class='wrap'><table>{''.join(trs)}</table></div>"
           f"</main></body></html>")
    Path(dest).write_text(doc, encoding="utf-8")
    if zip_path:
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for r in rows:
                z.write(r["wav"], Path(r["wav"]).name)
    return Path(dest)


def default_jobs(labels, mixes=(), seeds=(1, 2)):
    jobs = [dict(name=f"{l.replace('/', '__')}_{s}", label=l, seed=s) for l in labels for s in seeds]
    jobs += [dict(name="_plus_".join(m) + "_1", mix={k: 1.0 / len(m) for k in m}, seed=1) for m in mixes]
    return jobs


def recognizability(rows, labels, packs=None, device=None, n_real=8):
    """CLAP zero-shot: does each generated sound get classified as its label among `labels`? Real val clips of the
    same labels are scored the same way as the ceiling. Adds r["clap"] (predicted label) to the rows."""
    import soundfile as sf
    from .evaluate import Clap, category_match
    clap = Clap(device)
    gen_rows = [r for r in rows if r["spec"].get("label") in labels]
    wavs, srs = zip(*[sf.read(r["wav"], dtype="float32") for r in gen_rows]) if gen_rows else ((), ())
    res = dict(labels=labels)
    if gen_rows:
        m = category_match(clap, list(wavs), [r["spec"]["label"] for r in gen_rows], labels, sr=srs[0])
        for r, p in zip(gen_rows, m["predicted"]):
            r["clap"] = p
        res["generated_accuracy"] = m["accuracy"]
    for r in rows:
        if "clap" not in r and r["spec"].get("mix") is not None:
            mm = category_match(clap, [sf.read(r["wav"], dtype="float32")[0]], [labels[0]], labels,
                                sr=sf.info(r["wav"]).samplerate)
            r["clap"] = mm["predicted"][0]
    if packs:
        from .data import read_index
        real, real_l = [], []
        for p in packs:
            audio = np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r")
            sr = json.loads((Path(p) / "meta.json").read_text())["sample_rate"]
            cnt = {}
            for row in read_index(p):
                if row["split"] == "val" and row["label"] in labels and cnt.get(row["label"], 0) < n_real:
                    n = min(int(row["length"]), 4 * sr)
                    real.append(np.asarray(audio[int(row["offset"]): int(row["offset"]) + n], np.float32) / 32768)
                    real_l.append(row["label"])
                    cnt[row["label"]] = cnt.get(row["label"], 0) + 1
        if real:
            res["real_accuracy"] = category_match(clap, real, real_l, labels, sr=sr)["accuracy"]
    print(f"CLAP: generated {res.get('generated_accuracy', float('nan')):.2f} | "
          f"real {res.get('real_accuracy', float('nan')):.2f} (accuracy among {len(labels)} labels)")
    return res


KEPT = ["cough_sneeze", "dog_bark", "door_knock", "glass_break", "gunshot", "keyboard", "sea_waves", "thunder"]
MIXES = [("gunshot", "thunder"), ("glass_break", "door_knock"), ("dog_bark", "cough_sneeze")]


def run_and_publish(run_dir, vocoder, packs, work_dir, labels=None, mixes=None, seeds=2, hand_vectors=2,
                    guidance=2.0, clap=True, push=None):
    """Everything after training, in one call: sounds from scratch + novelty + CLAP + page + zip, copied to
    <run_dir>/publish/ and (push = colab.push_to_github) sent to GitHub results/<RUN>/. Returns the publish files."""
    import shutil
    run_dir, work = Path(run_dir), Path(work_dir)
    mg, _ = SketchDiffusion.load(run_dir / "mel_gen.pt")
    labels = [l for l in (labels or KEPT) if l in mg.vocab]
    mixes = [m for m in (mixes if mixes is not None else MIXES) if all(x in mg.vocab for x in m)]
    jobs = default_jobs(labels, mixes, seeds=tuple(range(1, seeds + 1)))
    E = mg.net.embed.weight.detach()[1:]
    g = torch.Generator().manual_seed(0)
    for i in range(hand_vectors):          # a random point around the category vectors, belongs to no category
        jobs.append(dict(name=f"vektor_{i + 1}", vec=E.mean(0) + E.std(0) * torch.randn(E.shape[1], generator=g),
                         seed=1))
    rows = from_scratch(run_dir / "mel_gen.pt", vocoder, jobs, work, packs=packs, guidance=guidance)
    if clap:
        try:
            rec = recognizability(rows, labels, packs=packs)
            (work / "clap.json").write_text(json.dumps(rec, indent=1))
        except Exception as e:                                   # noqa: BLE001  (CLAP download etc.)
            print(f"CLAP check skipped: {type(e).__name__}: {e}")
    name = run_dir.name
    page(rows, work / "index.html", f"{name} · sıfırdan üretim", zip_path=work / f"{name}_sesler.zip")
    pub = run_dir / "publish"
    pub.mkdir(exist_ok=True)
    ck = torch.load(run_dir / "mel_gen.pt", map_location="cpu", weights_only=False)
    ck["state"] = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["state"].items()}
    torch.save(ck, pub / "mel_gen.pt")
    for f in ("index.html", f"{name}_sesler.zip", "scratch.json", "clap.json"):
        if (work / f).exists():
            shutil.copy2(work / f, pub / f)
    if (run_dir / "loss.csv").exists():
        shutil.copy2(run_dir / "loss.csv", pub / "loss.csv")
    files = sorted(pub.iterdir())
    if push:
        push(files, dest=f"results/{name}", message=f"{name}: sounds from scratch (vector -> mel -> vocoder)",
             max_mb=95)
    return files
