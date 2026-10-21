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
from .sketch_model import SketchDiffusion, novelty


def _real_mels(packs, mel_cfg, frames, per_label=12, max_s=6.0, seed=0):
    from .audio import MelConfig, MelSketch
    from .data import read_index
    mel = MelSketch(MelConfig(**mel_cfg))
    rng = np.random.default_rng(seed)
    by_label = {}
    for p in packs:
        audio = np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r")
        rows = [r for r in read_index(p) if r["split"] == "train"]
        rng.shuffle(rows)
        for r in rows:
            lst = by_label.setdefault(r["label"], [])
            if len(lst) >= per_label:
                continue
            n = min(int(r["length"]), int(max_s * mel_cfg["sample_rate"]))
            a = np.asarray(audio[int(r["offset"]): int(r["offset"]) + n], np.float32) / 32768
            if len(a) >= (frames - 1) * mel_cfg["hop"]:
                lst.append(mel(torch.from_numpy(a)[None])[0].numpy())
    return by_label


def from_scratch(mel_gen, vocoder, jobs, out_dir, packs=None, steps=50, guidance=2.0, temperature=1.0,
                 device=None, per_label_ref=12):
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
    real = _real_mels(packs, gen.mel_cfg, mg.cfg.frames, per_label_ref) if packs else {}
    pool = [m for ms in real.values() for m in ms]
    # reference: distance between DIFFERENT real clips (same crop length as the generator)
    base = []
    rng = np.random.default_rng(1)
    for _ in range(min(16, len(pool))):
        i = int(rng.integers(len(pool)))
        m = pool[i]
        st = int(np.clip(np.argmax(m.mean(0)) - mg.cfg.frames // 3, 0, max(0, m.shape[1] - mg.cfg.frames)))
        base.append(novelty(m[:, st: st + mg.cfg.frames], pool[:i] + pool[i + 1:]))
    real_to_real = float(np.median(base)) if base else None
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
        nov = novelty(m, pool) if pool else None
        rows.append(dict(name=j["name"], wav=str(f), mel=m, edited=r["sketch"], seconds=len(a) / gen.sr,
                         spec={k: v for k, v in j.items() if k in ("label", "mix", "seed")},
                         nearest_db=nov, real_to_real_db=real_to_real))
        print(f"{j['name']:28s} {len(a) / gen.sr:.2f}s  nearest real {nov if nov is None else round(nov, 1)} dB"
              f" (real-to-real {real_to_real if real_to_real is None else round(real_to_real, 1)})")
    meta = [{k: v for k, v in r.items() if k not in ("mel", "edited")} for r in rows]
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
                   f"{sp.get('seed', 0)}</span></td><td><img alt='' src='{_png(r['mel'])}'></td>"
                   f"<td><audio controls preload='none' src='{_ogg(r['wav'])}'></audio></td></tr>")
    css = """:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e7e6e2}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 12px}
table{border-collapse:collapse;width:100%}td{border-top:1px solid var(--rule);padding:8px 6px;vertical-align:middle}
.lab{font-weight:600}.k{display:block;font-size:12px;font-weight:400;color:var(--ink2)}img{width:240px;height:auto;border-radius:4px;display:block}
audio{height:34px;max-width:260px}.wrap{overflow-x:auto}"""
    doc = (f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style>"
           f"</head><body><main><h1>{html.escape(title)}</h1><div class='wrap'><table>{''.join(trs)}</table></div>"
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
