"""One-folder demo of what the model can do (for listening and sharing).

    showcase(gen, "/content/showcase/base_v1")  ->  wavs + manifest.json + index.html

Sections: real vs generated for every category, category switching on one sketch,
parameter sweeps, and variations of a single request.
"""
import html
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .audio import SR, save_audio
from .controls import Controls

MAX_S = 4.0


def _save(folder, name, audio):
    save_audio(folder / f"{name}.wav", np.asarray(audio)[: int(MAX_S * SR)])
    return f"{name}.wav"


def showcase(gen, out_dir, seed=1, sweep_label=None, swap_labels=None, labels=None, refine=None,
             public_only=False):
    """public_only: real reference clips only when their license allows sharing (public repos)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = labels or gen.labels
    sections = []

    # 1. real vs generated (same coarse sketch -> the model invents the detail)
    items = []
    per_label = max(1, 4 // len(labels))            # a single-category model shows 4 pairs (bank keeps audio for 4)
    for label in tqdm(labels, desc="real vs generated", unit="label"):
        for k in range(per_label):
            r = gen.generate(label, Controls(), refine=refine, seed=seed, ref_index=k)
            name = label.replace("/", "__") + (f"_{k}" if per_label > 1 else "")
            row = dict(label=label, generated=_save(out, f"{name}_gen", r["audio"][0]))
            if r["reference_audio"] is not None and (r["reference_public_ok"] or not public_only):
                row["real"] = _save(out, f"{name}_real", r["reference_audio"])
                row["attribution"] = r.get("attribution") or ""
            elif r["reference_audio"] is not None:
                row["note"] = "gerçek kayıt paylaşılamaz lisanslı, gösterilmiyor"
            items.append(row)
    sections.append(dict(title="Gerçek ve üretilen", kind="pairs", items=items,
                         text="Model yalnız 20 bantlık kaba taslağı ve kategori adını görür; dalga formunun ayrıntısını kendisi üretir."))

    # 2. category switching: one sketch, different labels
    swap = swap_labels or [l for l in ("footsteps", "gunshot", "rain", "dog_bark", "car") if l in labels][:4]
    if len(swap) >= 2:
        ref = gen.reference(swap[0], seed=seed, index=0)
        base = ref["audio"].numpy() / 32768 if ref.get("audio") is not None else None
        items = []
        for label in tqdm(swap, desc="category switch", unit="label"):
            r = gen.generate(label, Controls(), refine=refine, seed=seed, reference=base) if base is not None else \
                gen.generate(label, Controls(), refine=refine, seed=seed)
            items.append(dict(label=label, generated=_save(out, f"switch_{label.replace('/', '__')}", r["audio"][0])))
        sections.append(dict(title=f"Kategori geçişi: aynı taslak ({swap[0]}), farklı kategori", kind="list",
                             items=items, text="Aynı zaman/enerji taslağı, kategori etiketi değişince farklı bir sese dönüşür."))

    # 3. parameter sweeps
    probe = sweep_label or next((l for l in ("footsteps", "gunshot", "rain") if l in labels), labels[0])
    sweeps = [("intensity", [-1.0, 0.0, 1.0]), ("distance", [0.0, 0.5, 1.0]),
              ("environment", ["dry", "room", "hall", "cave"]), ("speed", [0.7, 1.0, 1.5]),
              ("weight", [-1.0, 0.0, 1.0]), ("duration", [1.5, 4.0])]
    for param, values in tqdm(sweeps, desc=f"parameter sweeps ({probe})", unit="param"):
        items = []
        for v in values:
            r = gen.generate(probe, Controls(**{param: v}), refine=refine, seed=seed, ref_index=0)
            items.append(dict(label=f"{param} = {v}",
                              generated=_save(out, f"sweep_{param}_{str(v).replace('.', 'p').replace('-', 'm')}", r["audio"][0])))
        sections.append(dict(title=f"Parametre: {param} ({probe})", kind="list", items=items, text=""))

    # 4. variations of one request
    items = []
    for i, (s, mc) in enumerate([(seed, False), (seed + 1, False), (seed + 2, False), (seed, True)]):
        r = gen.generate(probe, Controls(), refine=refine, seed=s, mc_dropout=mc, ref_index=0)
        items.append(dict(label=f"seed {s}" + (" + MC dropout" if mc else ""),
                          generated=_save(out, f"variation_{i}", r["audio"][0])))
    sections.append(dict(title=f"Varyasyonlar: aynı istek ({probe})", kind="list", items=items,
                         text="Aynı kategori ve parametreler; farklı gürültü tohumu ve MC dropout ile farklı örnekler."))

    manifest = dict(model_step=gen.step, labels=labels, sections=sections)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    (out / "index.html").write_text(_html(manifest), encoding="utf-8")
    files = sorted(out.glob("*"))
    print(f"showcase: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB -> {out}")
    return files


def _html(m):
    rows = []
    for sec in m["sections"]:
        rows.append(f"<h2>{html.escape(sec['title'])}</h2><p>{html.escape(sec['text'])}</p><table>")
        for it in sec["items"]:
            cells = [f"<td>{html.escape(it['label'])}</td>"]
            for key in ("real", "generated"):
                if key in it:
                    cells.append(f"<td>{key}<br><audio controls src='{it[key]}'></audio></td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        rows.append("</table>")
    return ("<!doctype html><meta charset='utf-8'><title>sound-gen showcase</title>"
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto}td{padding:6px}</style>"
            f"<h1>sound-gen · adım {m['model_step']}</h1>" + "\n".join(rows))
