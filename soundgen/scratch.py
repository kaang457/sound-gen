"""Sounds from scratch: vector -> mel generator -> (controls) -> vocoder -> audio. No reference sound.

    rows = from_scratch("runs/melgen_v1/mel_gen.pt", "bigvgan", jobs, "out/", packs=[pack])
    page(rows, "out/index.html", "melgen_v1 · sıfırdan")

A job is a dict:  name, and ONE of  label="gunshot" | text="spaceship engine hum" (CLAP models) |
mix={"gunshot": .7, "thunder": .3} (keys: labels, or free text for CLAP models) | vec=<tensor [vec_dim]>,
plus optional seed, controls (Controls), guidance, frames. The vocoder is "bigvgan" or a WaveFlow checkpoint.

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
from .sketch_model import load_generator


def _clip_mels(packs, mel_cfg, split, per_label, max_s, seed, device):
    """{label: [(clip_id, mel [n_mels, F] on device)]} from a pack split."""
    from .audio import MelSketch
    from .data import read_index, pack_sample_rate
    rng = np.random.default_rng(seed)
    out = {}
    for p in packs:
        src = pack_sample_rate(p)
        mel = MelSketch(mel_cfg, in_sr=src).to(device)
        audio = np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r")
        rows = [r for r in read_index(p) if r["split"] == split]
        rng.shuffle(rows)
        for r in rows:
            lst = out.setdefault(r["label"], [])
            if per_label and len(lst) >= per_label:
                continue
            n = min(int(r["length"]), int(max_s * src))
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


def from_scratch(mel_gen, vocoder, jobs, out_dir, packs=None, steps=50, guidance=2.0, device=None,
                 per_label_ref=150):
    from .audio import save_audio, mel_config
    from .vocoder import load_vocoder, render
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mg, ck = load_generator(mel_gen, device)
    voc = load_vocoder(vocoder, device)
    if ck.get("mel_cfg") and mel_config(ck["mel_cfg"]) != mel_config(voc.mel_cfg):
        raise ValueError(f"mel contract mismatch: generator {ck['mel_cfg']} vs vocoder {voc.mel_cfg}")
    if getattr(voc, "vocoder", True) is False:
        print("note: WaveFlow checkpoint is category-conditioned, not a universal vocoder")
    index = NoveltyIndex(packs, voc.mel_cfg, mg.cfg.frames, per_label=per_label_ref, device=device) if packs else None
    real_to_real = index.baseline() if index else None
    if index:
        print(f"novelty index: {index.n} training clips; unseen real recordings are {real_to_real:.1f} dB "
              f"from their nearest training clip (median)")
    rows = []
    for j in jobs:
        seed = j.get("seed", 0)
        if "vec" in j:
            vec = torch.as_tensor(j["vec"], dtype=torch.float32)
        elif j.get("text"):
            vec = mg.vector(text=j["text"])
        else:
            vec = mg.vector(label=j.get("label"), mix=j.get("mix"))
        g = torch.Generator(device=device).manual_seed(seed)
        m = mg.sample(vec, n=1, steps=steps, guidance=j.get("guidance", guidance), frames=j.get("frames"),
                      generator=g)[0].cpu().numpy()
        top = j.get("label") or (max(j["mix"], key=j["mix"].get) if j.get("mix") else None)
        a, edited = render(voc, m, j.get("controls") or Controls(), _kind(top), seed)
        if j.get("fx") is not None:                       # the studio's effects rack, applied after the vocoder
            from .fx import process
            a = process(a, voc.sr, j["fx"])[1]
        f = out / f"{j['name']}.wav"
        save_audio(f, a, voc.sr)
        nov, nl, nid = index.nearest(m) if index else (None, None, None)
        rows.append(dict(name=j["name"], wav=str(f), mel=m, edited=edited, seconds=len(a) / voc.sr,
                         spec={k: v for k, v in j.items() if k in ("label", "mix", "seed", "text", "note")},
                         controls=(j["controls"].to_dict() if j.get("controls") else None),
                         fx=(j["fx"].to_dict() if j.get("fx") is not None else None),
                         events=count_events(a, voc.sr),
                         nearest_db=nov, nearest_label=nl, nearest_id=nid, real_to_real_db=real_to_real))
        print(f"{j['name']:28s} {len(a) / voc.sr:.2f}s  events {rows[-1]['events']:2d}  nearest training clip "
              f"{'-' if nov is None else f'{nov:.1f} dB ({nl})'}  | unseen real: "
              f"{'-' if real_to_real is None else f'{real_to_real:.1f} dB'}")
    meta = [{k: v for k, v in r.items() if k not in ("mel", "edited")} for r in rows]
    meta.append(dict(real_to_real_db=real_to_real, index_clips=index.n if index else 0, vocoder=voc.kind))
    (out / "scratch.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return rows


def _kind(label):
    """Controls treat textures (loopable) and events differently; unknown labels / prompts count as events."""
    if not label:
        return "event"
    try:
        from .data import load_categories
        cats = load_categories()["categories"]
        return cats.get(label, cats.get(label.split("/")[0], {})).get("type", "event")
    except Exception:                                            # noqa: BLE001
        return "event"


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
        what = (f"“{sp['text']}”" if sp.get("text") else sp.get("label")
                or " + ".join(f"{k} {v:.0%}" for k, v in (sp.get("mix") or {}).items()) or "elle vektör")
        if sp.get("note"):
            what += f" · {sp['note']}"
        trs.append(f"<tr><td class='lab'>{html.escape(r['name'])}<span class='k'>{html.escape(what)} · tohum "
                   f"{sp.get('seed', 0)}</span>"
                   + (f"<span class='k'>en yakın eğitim kaydı: {r['nearest_db']:.1f} dB ({html.escape(str(r['nearest_label']))})</span>"
                      if r.get("nearest_db") is not None else "")
                   + (f"<span class='k'>olay sayısı ≈ {r['events']}</span>" if r.get("events") is not None else "")
                   + (f"<span class='k'>CLAP: {html.escape(r['clap'])}</span>" if r.get("clap") else "")
                   + (f"<span class='k'>istem eşleşmesi (CLAP): {'✓' if r['prompt_ok'] else '✗'} "
                      f"(benzerlik {r['prompt_sim']:.2f})</span>" if r.get("prompt_sim") is not None else "")
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
    """Evaluation CLAP (laion/clap-htsat-unfused, NOT the conditioning model, so the check is less circular).
    1. category jobs: zero-shot class among `labels`, accuracy vs the same score on unseen real clips (val);
    2. prompt jobs: is the generated sound closer to its own prompt than to the other prompts and the labels?
    Adds r["clap"], r["prompt_sim"], r["prompt_ok"] to the rows."""
    import soundfile as sf
    from .evaluate import Clap, category_match, label_text
    clap = Clap(device)
    res = dict(labels=labels)
    gen_rows = [r for r in rows if r["spec"].get("label") in labels and not r.get("controls") and not r.get("fx")]
    if gen_rows:
        wavs = [sf.read(r["wav"], dtype="float32")[0] for r in gen_rows]
        m = category_match(clap, wavs, [r["spec"]["label"] for r in gen_rows], labels, sr=sf.info(gen_rows[0]["wav"]).samplerate)
        for r, p in zip(gen_rows, m["predicted"]):
            r["clap"] = p
        res["generated_accuracy"] = m["accuracy"]
    for r in rows:
        if "clap" not in r and r["spec"].get("mix") is not None and labels:
            mm = category_match(clap, [sf.read(r["wav"], dtype="float32")[0]], [labels[0]], labels,
                                sr=sf.info(r["wav"]).samplerate)
            r["clap"] = mm["predicted"][0]
    prompt_rows = [r for r in rows if r["spec"].get("text")]
    if prompt_rows:
        prompts = sorted({r["spec"]["text"] for r in prompt_rows})
        cands = prompts + [label_text(l) for l in labels]
        t = clap.text(cands)
        ok = []
        for r in prompt_rows:
            x, sr = sf.read(r["wav"], dtype="float32")
            a = clap.audio([x], sr)[0]
            sims = t @ a
            own = cands.index(r["spec"]["text"])
            r["prompt_sim"], r["prompt_ok"] = float(sims[own]), bool(int(sims.argmax()) == own)
            ok.append(r["prompt_ok"])
        res["prompt_accuracy"] = float(np.mean(ok))
        res["prompt_candidates"] = len(cands)
    if packs and labels:
        from .data import read_index, pack_sample_rate
        real, real_l = [], []
        for p in packs:
            audio = np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r")
            sr = pack_sample_rate(p)
            cnt = {}
            for row in read_index(p):
                if row["split"] == "val" and row["label"] in labels and cnt.get(row["label"], 0) < n_real:
                    n = min(int(row["length"]), 4 * sr)
                    real.append(np.asarray(audio[int(row["offset"]): int(row["offset"]) + n], np.float32) / 32768)
                    real_l.append(row["label"])
                    cnt[row["label"]] = cnt.get(row["label"], 0) + 1
        if real:
            res["real_accuracy"] = category_match(clap, real, real_l, labels, sr=sr)["accuracy"]
    print(f"CLAP: categories generated {res.get('generated_accuracy', float('nan')):.2f} | real "
          f"{res.get('real_accuracy', float('nan')):.2f} (among {len(labels)} labels) · prompts matched "
          f"{res.get('prompt_accuracy', float('nan')):.2f} (among {res.get('prompt_candidates', 0)} candidates)")
    return res


KEPT = ["cough_sneeze", "dog_bark", "door_knock", "glass_break", "gunshot", "keyboard", "sea_waves", "thunder"]
MIXES = [("gunshot", "thunder"), ("glass_break", "door_knock"), ("dog_bark", "cough_sneeze")]
PROMPTS = ["a single gunshot", "a gunfight with many gunshots", "a dog barking", "many dogs barking",
           "spaceship engine hum", "sci-fi laser gun shot", "metal sword clash", "monster growl",
           "distant explosion", "heavy rain on a tin roof"]
# pairs whose difference the model must understand (one event vs many): compared on the page and in clap.json
PAIRS = [("a single gunshot", "a gunfight with many gunshots"), ("a dog barking", "many dogs barking")]


def demo_jobs(mg, labels=None, mixes=None, prompts=None, seeds=2, hand_vectors=2, controls_demo=True):
    """The standard from-scratch test set: categories x seeds, mixes, prompts (CLAP models), hand-typed vectors,
    and the controls applied to a generated gunshot (firefight, distance, cave)."""
    from .controls import Controls
    labels = [l for l in (labels or KEPT) if l in mg.vocab]
    mixes = [m for m in (mixes if mixes is not None else MIXES) if all(x in mg.vocab for x in m)]
    jobs = default_jobs(labels, mixes, seeds=tuple(range(1, seeds + 1)))
    clap_model = getattr(mg, "label_vecs", None) is not None
    if clap_model:
        for k, ptxt in enumerate(prompts if prompts is not None else PROMPTS):
            jobs.append(dict(name=f"istem_{k + 1}", text=ptxt, seed=1))
        V = mg.label_vecs.detach()[1:]
    else:
        V = mg.net.embed.weight.detach()[1:]
    g = torch.Generator().manual_seed(0)
    for i in range(hand_vectors):          # a random point around the category vectors, belongs to no category
        v = V.mean(0).cpu() + V.std(0).cpu() * torch.randn(V.shape[1], generator=g)
        if clap_model:
            v = v / v.norm()
        jobs.append(dict(name=f"vektor_{i + 1}", vec=v, seed=1))
    if controls_demo and "gunshot" in mg.vocab:            # the studio's effects rack on a generated gunshot
        from .fx import Rack
        for name, rk, note in [("uzak", Rack(distance=0.8), "efekt: uzakta"),
                               ("magara", Rack(environment="cave"), "efekt: mağarada"),
                               ("uzak_salon_sol", Rack(distance=0.5, environment="hall", pan=-0.6), "efekt: uzak, salon")]:
            jobs.append(dict(name=f"gunshot_{name}", label="gunshot", seed=1, fx=rk, note=note))
    return jobs, labels


def run_and_publish(run_dir, vocoder, packs, work_dir, labels=None, mixes=None, prompts=None, seeds=2,
                    hand_vectors=2, guidance=2.0, clap=True, push=None):
    """Everything after training, in one call: sounds from scratch + novelty + CLAP + page + zip, copied to
    <run_dir>/publish/ and (push = colab.push_to_github) sent to GitHub results/<RUN>/. Returns the publish files."""
    import shutil
    run_dir, work = Path(run_dir), Path(work_dir)
    mg, _ = load_generator(run_dir / "mel_gen.pt")
    jobs, labels = demo_jobs(mg, labels, mixes, prompts, seeds, hand_vectors)
    rows = from_scratch(run_dir / "mel_gen.pt", vocoder, jobs, work, packs=packs, guidance=guidance)
    if clap:
        try:
            rec = recognizability(rows, labels, packs=packs)
            (work / "clap.json").write_text(json.dumps(rec, indent=1))
        except Exception as e:                                   # noqa: BLE001  (CLAP download etc.)
            print(f"CLAP check skipped: {type(e).__name__}: {e}")
    by_text = {r["spec"].get("text"): r for r in rows if r["spec"].get("text")}
    pairs = [dict(one=a, many=b, events_one=by_text[a]["events"], events_many=by_text[b]["events"])
             for a, b in PAIRS if a in by_text and b in by_text]
    for p_ in pairs:
        print(f"one vs many: '{p_['one']}' {p_['events_one']} events | '{p_['many']}' {p_['events_many']} events")
    (work / "pairs.json").write_text(json.dumps(pairs, indent=1))
    name = run_dir.name
    page(rows, work / "index.html", f"{name} · sıfırdan üretim", zip_path=work / f"{name}_sesler.zip")
    pub = run_dir / "publish"
    pub.mkdir(exist_ok=True)
    ck = torch.load(run_dir / "mel_gen.pt", map_location="cpu", weights_only=False)
    ck["state"] = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["state"].items()}
    torch.save(ck, pub / "mel_gen.pt")
    for f in ("index.html", f"{name}_sesler.zip", "scratch.json", "clap.json", "pairs.json"):
        if (work / f).exists():
            shutil.copy2(work / f, pub / f)
    if (run_dir / "loss.csv").exists():
        shutil.copy2(run_dir / "loss.csv", pub / "loss.csv")
    files = sorted(pub.iterdir())
    if push:
        push(files, dest=f"results/{name}", message=f"{name}: sounds from scratch (vector -> mel -> vocoder)",
             max_mb=95)
    return files


def count_events(x, sr, floor_db=-40.0, min_gap=0.06, prominence_db=8.0):
    """Rough number of distinct events (onsets) in a sound: peaks of the 10 ms energy envelope that stand
    prominence_db above their surroundings, within floor_db of the loudest, at least min_gap apart."""
    from scipy.signal import find_peaks
    x = np.asarray(x, np.float32)
    hop = int(0.01 * sr)
    if len(x) < hop * 5:
        return 0
    n = len(x) // hop
    e = 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    e = np.convolve(e, np.ones(3) / 3, "same")
    pk, _ = find_peaks(e, height=e.max() + floor_db, prominence=prominence_db, distance=max(1, int(min_gap / 0.01)))
    return int(len(pk))
