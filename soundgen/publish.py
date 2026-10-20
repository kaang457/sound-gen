"""One tidy folder per training run, small enough for GitHub.

    publish_run(run_dir, packs)  ->  <run_dir>/publish/
        README.md        summary: step, losses, data, categories, how to use
        model.pt         EMA weights in fp16 (what Generator uses), ~45 MB for 128 channels
        sketch_bank.pt   reference sketches; real audio kept only where the license allows sharing
        showcase.html    every sound embedded (OGG): real vs generated, sweeps, category switch, variations
        health.html      training health report

Everything else (checkpoints, per-step samples, logs) stays on Drive.
"""
import base64
import html
import io
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .audio import SR

GITHUB_FILE_LIMIT_MB = 95          # GitHub refuses files over 100 MB


def slim_model(ckpt, dest):
    """EMA weights only, floats in fp16. Loads with Generator / load_model like a full checkpoint."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ema = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["ema"].items()}
    out = dict(ema=ema, flow_cfg=ck["flow_cfg"], mel_cfg=ck["mel_cfg"], vocab=ck["vocab"],
               step=ck.get("step"), best=ck.get("best"), train_cfg=ck.get("train_cfg"))
    torch.save(out, dest)
    return Path(dest)


def public_bank(src, dest):
    """Sketch bank with reference audio removed wherever the clip may not be redistributed."""
    bank = torch.load(src, map_location="cpu", weights_only=False)
    removed = 0
    for items in bank["items"].values():
        for it in items:
            if it.get("audio") is not None and not it.get("public_ok"):
                it["audio"] = None
                removed += 1
    torch.save(bank, dest)
    return Path(dest), removed


def _ogg_data_uri(wav_path):
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="OGG", subtype="VORBIS")
    return "data:audio/ogg;base64," + base64.b64encode(buf.getvalue()).decode()


def bundle_showcase(folder, dest, title):
    """showcase() folder (wavs + manifest.json) -> one self-contained HTML page."""
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text())
    parts = []
    for sec in m["sections"]:
        rows = []
        for it in sec["items"]:
            cells = [f"<td class='lab'>{html.escape(it['label'])}</td>"]
            for key, name in (("real", "gerçek"), ("generated", "üretilen")):
                if key in it:
                    cells.append(f"<td><span class='k'>{name}</span><audio controls preload='none' "
                                 f"src='{_ogg_data_uri(folder / it[key])}'></audio></td>")
            if it.get("note"):
                cells.append(f"<td class='note'>{html.escape(it['note'])}</td>")
            if it.get("attribution"):
                cells.append(f"<td class='note'>{html.escape(it['attribution'])}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        parts.append(f"<section><h2>{html.escape(sec['title'])}</h2>"
                     + (f"<p>{html.escape(sec['text'])}</p>" if sec.get("text") else "")
                     + f"<div class='wrap'><table>{''.join(rows)}</table></div></section>")
    page = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root{{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e7e6e2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px 48px}} h1{{font-size:22px;margin:0 0 4px}}
h2{{font-size:17px;margin:32px 0 4px}} p,.note,.k{{color:var(--ink2)}} p{{margin:0 0 8px}}
table{{border-collapse:collapse;width:100%}} td{{border-top:1px solid var(--rule);padding:8px 6px;vertical-align:middle}}
.lab{{font-weight:600;white-space:nowrap}} .k{{display:block;font-size:12px}} .note{{font-size:12px}}
audio{{height:34px;max-width:260px}} .wrap{{overflow-x:auto}}
</style></head><body><main><h1>{html.escape(title)}</h1>
<p>Model adım {m.get('model_step')} · kategoriler: {html.escape(', '.join(m.get('labels', [])))}</p>
{''.join(parts)}</main></body></html>"""
    Path(dest).write_text(page, encoding="utf-8")
    return Path(dest)


def _readme(run, gen, packs, refine, removed, sizes):
    from .diagnostics import read_log
    train, val, _ = read_log(run, with_grad=True)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    tc = ck.get("train_cfg") or {}
    rows, lic = Counter(), Counter()
    hours = 0.0
    for p in packs:
        meta = json.loads((Path(p) / "meta.json").read_text())
        hours += meta.get("hours", 0)
        import csv
        for r in csv.DictReader(open(Path(p) / "index.csv")):
            rows[r["label"]] += 1
            lic[r["license"].split(" (")[0]] += 1
    best_val = min((v for _, v in val), default=float("nan"))
    lines = [
        f"# {run.name}", "",
        f"- Adım: **{ck.get('step')}** · en iyi doğrulama NLL: **{best_val:.4f}** (düşük = iyi)",
        f"- Veri: {hours:.1f} saat, {sum(rows.values())} parça, {len(rows)} etiket",
        f"- Model: {ck['flow_cfg'].get('n_flows')} flow × {ck['flow_cfg'].get('n_layers')} katman, "
        f"{ck['flow_cfg'].get('res_channels')} kanal, upsampler `{ck['flow_cfg'].get('upsampler', 'transposed')}`, "
        f"cond_norm {ck['flow_cfg'].get('cond_norm', False)}, actnorm {ck['flow_cfg'].get('actnorm', False)}",
        f"- Vitrin son işlemi: `refine={refine!r}`", "",
        "| Dosya | Ne |", "|---|---|",
        "| `showcase.html` | Bütün sesler tek sayfada (indirip tarayıcıda aç) |",
        "| `health.html` | Eğitim sağlık raporu |",
        f"| `model.pt` | Üretim ağırlıkları (EMA, fp16), {sizes.get('model.pt', 0):.0f} MB |",
        f"| `sketch_bank.pt` | Referans taslaklar; paylaşılamaz lisanslı {removed} gerçek kayıt çıkarıldı |", "",
        "## Kullanım", "", "```python",
        "from soundgen.generate import Generator",
        "from soundgen.controls import Controls",
        f"gen = Generator('results/{run.name}/model.pt')   # sketch_bank.pt aynı klasörden okunur",
        "out = gen.generate('gunshot', Controls(distance=0.5, environment='hall'), refine='v1')",
        "```", "",
        "## Etiketler (parça sayısı)", "",
        ", ".join(f"{k} ({v})" for k, v in rows.most_common()), "",
        "## Lisanslar (parça sayısı)", "",
        ", ".join(f"{k}: {v}" for k, v in lic.most_common()), "",
    ]
    if any("NC" in k or "BBC" in k or "research" in k.lower() for k in lic):
        lines += ["**Uyarı:** Eğitim verisinde ticari kullanıma izin vermeyen lisanslar var; bu modelin çıktıları "
                  "ticari olarak kullanılamaz (yalnız araştırma / kavram kanıtı).", ""]
    return "\n".join(lines)


def publish_run(run_dir, packs, refine="v1", ckpt="best.pt", health=True):
    """Build <run_dir>/publish/ and return its files (push them with colab.push_to_github)."""
    from .generate import Generator
    from .showcase import showcase
    run = Path(run_dir)
    out = run / "publish"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = []

    # model + bank (the showcase is generated from exactly what is published)
    model = slim_model(run / ckpt, out / "model.pt")
    bank, removed = public_bank(run / "sketch_bank.pt", out / "sketch_bank.pt")
    gen = Generator(run / ckpt, bank=run / "sketch_bank.pt")
    with tempfile.TemporaryDirectory() as tmp:
        showcase(gen, tmp, refine=refine, public_only=True)
        files.append(bundle_showcase(tmp, out / "showcase.html", f"{run.name} · vitrin"))
    if health:
        try:
            from .diagnostics import report
            _, path = report(run, packs=[str(p) for p in packs], ckpt="last.pt")
            files.append(Path(shutil.copy2(path, out / "health.html")))
        except Exception as e:                       # noqa: BLE001
            print(f"health report skipped: {type(e).__name__}: {e}")
    sizes = {}
    for f in (model, bank):
        mb = f.stat().st_size / 1e6
        if mb > GITHUB_FILE_LIMIT_MB:
            print(f"{f.name}: {mb:.0f} MB > {GITHUB_FILE_LIMIT_MB} MB, GitHub'a gönderilmeyecek (Drive'da: {f})")
        else:
            files.append(f)
            sizes[f.name] = mb
    (out / "README.md").write_text(_readme(run, gen, packs, refine, removed, sizes), encoding="utf-8")
    files.append(out / "README.md")
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"publish: {len(files)} files, {total:.1f} MB -> {out}")
    return sorted(files)
