"""Fine-tune Stable Audio 3 Small-SFX with LoRA on our own audio + captions, then compare against the stock models.

    from soundgen import sa3_lora
    stats = sa3_lora.prepare([P.drive / "lora_audio"], "/content/lora_data")          # local copy, captions prefixed
    ckpt = sa3_lora.train("/content/lora_data", P.drive / "sa3_lora" / RUN, steps=5000)
    sa3_lora.compare(ckpt, P.runs / RUN, push=colab.push_to_github)                    # stock vs base vs base + LoRA

Uses Stability's own trainer (scripts/train_lora.py at the pinned commit): LoRA / DoRA adapters on the -base
checkpoint (the trainer accepts only base models), frozen weights in bf16, only the adapter (~50-200 MB) is saved.
Inference with the LoRA uses the base model (50 steps, guidance 7): soundgen.sa3.SA3("small-sfx-base", lora=ckpt).
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .sa3 import PREFIX

AUDIO = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".aif")


# ------------------------------------------------------------------------------------------------ data
def prepare(src_dirs, dst, prefix=PREFIX, min_seconds=0.5, min_caption_words=2, sources=None, max_per_source=None,
            dedupe=True):
    """Copy audio + caption pairs (from soundgen.sources stores or any audio/.txt folders) to a local folder for
    training: drops files without a usable caption, prefixes captions like inference does, removes byte-identical
    duplicates (the Freesound-based sets overlap). -> stats dict"""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    seen, stats = set(), {}
    for src in [Path(s) for s in src_dirs]:
        man = {}
        if (src / "manifest.csv").exists():
            for r in csv.DictReader(open(src / "manifest.csv", newline="", encoding="utf-8")):
                man[r["path"]] = r
        for a in sorted(p for p in src.rglob("*") if p.suffix.lower() in AUDIO):
            rel = str(a.relative_to(src))
            source = man.get(rel, {}).get("source") or a.parent.name
            if sources and source not in sources:
                continue
            st = stats.setdefault(source, dict(files=0, seconds=0.0, skipped=0))
            if max_per_source and st["files"] >= max_per_source:
                continue
            txt = a.with_suffix(".txt")
            cap = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
            cap = re.sub(r"\s+", " ", cap)
            secs = float(man.get(rel, {}).get("seconds") or 0) or _duration(a) or 0
            if len(cap.split()) < min_caption_words or secs < min_seconds:
                st["skipped"] += 1
                continue
            if dedupe:
                h = _quick_hash(a)
                if h in seen:
                    st["skipped"] += 1
                    continue
                seen.add(h)
            out = dst / source / a.name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copy2(a, out)
            if prefix and not cap.startswith(prefix.strip()):
                cap = prefix + cap
            out.with_suffix(".txt").write_text(cap, encoding="utf-8")
            st["files"] += 1
            st["seconds"] += secs
    total = dict(files=sum(v["files"] for v in stats.values()), hours=round(sum(v["seconds"] for v in stats.values()) / 3600, 2))
    for k, v in sorted(stats.items()):
        print(f"{k:16s} {v['files']:7d} dosya {v['seconds'] / 3600:7.2f} saat  ({v['skipped']} atlandı)")
    print(f"{'TOPLAM':16s} {total['files']:7d} dosya {total['hours']:7.2f} saat -> {dst}")
    (dst / "stats.json").write_text(json.dumps(dict(total=total, by_source=stats), indent=1))
    return dict(total=total, by_source=stats)


def _quick_hash(p, n=1 << 16):
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(n))
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def _duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------ train
def checkpoints(save_dir):
    """All LoRA checkpoints under save_dir, oldest first: (part, step, path)."""
    out = []
    for p in Path(save_dir).rglob("*.ckpt"):
        m = re.search(r"step=(\d+)", p.name)
        part = re.search(r"part(\d+)", str(p.parent))
        out.append((int(part.group(1)) if part else 0, int(m.group(1)) if m else 0, p))
    return sorted(out)


def done_steps(save_dir):
    """Total steps over all parts (each resumed part starts counting from 0 again)."""
    last = {}
    for part, step, _ in checkpoints(save_dir):
        last[part] = max(last.get(part, 0), step)
    return sum(last.values())


def latest(save_dir):
    c = checkpoints(save_dir)
    return c[-1][2] if c else None


def auto_batch():
    import torch
    if not torch.cuda.is_available():
        return 1
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 32 if gb >= 70 else 16 if gb >= 35 else 8 if gb >= 20 else 4


def train(data_dir, save_dir, model="small-sfx-base", steps=5000, batch=None, duration=10.0, rank=16,
          adapter="dora-rows", lr=1e-4, workers=4, checkpoint_every=500, max_minutes=None, resume=True):
    """Run Stability's LoRA trainer; resumable (continues from the latest checkpoint in save_dir, in a new part
    folder, until `steps` in total). Stops cleanly after max_minutes (keeps the last checkpoint). -> latest ckpt"""
    import torch
    from . import sa3
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Bu GPU bf16 desteklemiyor (T4). Eğitim için A100 ya da L4 seç.")
    script = sa3.install_train()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    have = done_steps(save_dir) if resume else 0
    if have >= steps:
        print(f"zaten {have} adım eğitilmiş (hedef {steps}); son: {latest(save_dir)}")
        return latest(save_dir)
    parts = {p for p, _, _ in checkpoints(save_dir)}
    part_dir = save_dir / f"part{(max(parts) + 1) if parts else 0}"
    batch = batch or auto_batch()
    # checkpoints are written every `checkpoint_every` steps only: round the part up so its last step is saved
    todo = -(-(steps - have) // checkpoint_every) * checkpoint_every
    cmd = [sys.executable, str(script), "--model", model, "--data_dir", str(data_dir), "--save_dir", str(part_dir),
           "--steps", str(todo), "--batch_size", str(batch), "--duration", str(duration),
           "--rank", str(rank), "--adapter_type", adapter, "--lr", str(lr), "--num_workers", str(workers),
           "--checkpoint_every", str(checkpoint_every), "--log_every", "50", "--demo_every", str(10**9),
           "--logger", "csv", "--base_precision", "bf16"]
    prev = latest(save_dir) if resume else None
    if prev:
        cmd += ["--lora_checkpoint", str(prev)]
    (save_dir / "config.json").write_text(json.dumps(dict(model=model, steps=steps, batch=batch, duration=duration,
                                                          rank=rank, adapter=adapter, lr=lr, data_dir=str(data_dir),
                                                          resumed_from=str(prev) if prev else None), indent=1))
    print(f"eğitim: {model}, {todo} adım (toplam hedef {steps}, yapılmış {have}), batch {batch}, "
          f"{duration:g} s kırpma, {adapter} rank {rank}" + (f", devam: {prev.name}" if prev else ""), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_MININTERVAL="30")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(Path(script).parents[1]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    last_print = 0.0
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            progress = "it/s" in line or "s/it" in line
            if not progress or time.time() - last_print > 60:          # progress bar: once a minute
                print(line[-300:], flush=True)
                last_print = time.time() if progress else last_print
            if max_minutes and time.time() - t0 > max_minutes * 60:
                print(f"süre doldu ({max_minutes} dk): durduruluyor, son kayıt korunuyor", flush=True)
                proc.terminate()
                break
        proc.wait(timeout=120)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    ck = latest(save_dir)
    print(f"bitti: {(time.time() - t0) / 60:.0f} dk, toplam {done_steps(save_dir)} adım, son kayıt: {ck}")
    if proc.returncode not in (0, None, -15) and ck is None:
        raise RuntimeError(f"eğitim hata ile bitti (kod {proc.returncode}); yukarıdaki çıktıya bak")
    return ck


def loss_curve(save_dir):
    """CSV logger output -> (steps, losses) (best effort)."""
    xs, ys = [], []
    for f in sorted(Path(save_dir).rglob("metrics.csv")):
        for r in csv.DictReader(open(f)):
            k = next((k for k in r if "loss" in k and r[k]), None)
            if k and r.get("step"):
                xs.append(int(float(r["step"])))
                ys.append(float(r[k]))
    return xs, ys


# ------------------------------------------------------------------------------------------------ compare
def compare(lora, run_dir, items=None, n=2, seed=1, push=None, strength=1.0,
            variants=("small-sfx", "small-sfx-base", "small-sfx-base+lora")):
    """Same prompts through each model: stock post-trained, stock base, base + our LoRA. CLAP prompt match per
    variant; one page per variant + a summary README -> results/<run>/ (push)."""
    import gc
    import torch
    from . import sa3
    run_dir = Path(run_dir)
    items = items or sa3.REFERENCE
    summary, files = [], []
    for v in variants:
        model = v.split("+")[0]
        gen = sa3.SA3(model, lora=lora if v.endswith("+lora") else None, lora_strength=strength)
        out = sa3.reference_run(gen, run_dir / v.replace("+", "_"), items=items, n=n, seed=seed, clap=True)
        m = json.loads((run_dir / v.replace("+", "_") / "publish" / "measures.json").read_text())
        c = m["info"].get("clap") or {}
        summary.append(dict(variant=v, top1=c.get("top1"), top3=c.get("top3"), ranks={
            s["short"]: [x.get("clap_rank") for x in s["vars"]] for s in m["sounds"]}))
        files += [(v, f) for f in out]
        del gen
        gc.collect()
        torch.cuda.empty_cache()
    lines = [f"# {run_dir.name}: Stable Audio 3 LoRA karşılaştırması", "",
             f"LoRA: `{Path(str(lora)).name}` · güç {strength} · istek başı {n} varyasyon. Her modelin sayfası kendi "
             "klasöründe (`index.html`).", "",
             "| Model | İsteği 1. sırada | İlk 3'te |", "|---|---|---|"]
    for s in summary:
        f = lambda x: "–" if x is None else f"%{100 * x:.0f}"
        lines.append(f"| {s['variant']} | {f(s['top1'])} | {f(s['top3'])} |")
    shorts = list(summary[0]["ranks"])
    lines += ["", "| Ses | " + " | ".join(s["variant"] for s in summary) + " |",
              "|---|" + "---|" * len(summary)]
    for k in shorts:
        lines.append(f"| {k} | " + " | ".join(" ".join(str(r) for r in s["ranks"].get(k, [])) for s in summary) + " |")
    lines += ["", "Sayılar: CLAP sırası (1 = ses, bütün istekler arasında en çok kendi isteğine benziyor). Dinlemenin "
              "yerini tutmaz."]
    readme = run_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "compare.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("\n".join(lines[:8 + len(summary)]))
    if push:
        push([readme, run_dir / "compare.json"], dest=f"results/{run_dir.name}",
             message=f"{run_dir.name}: LoRA comparison")
        for v in variants:
            pub = run_dir / v.replace("+", "_") / "publish"
            push(sorted(pub.iterdir()), dest=f"results/{run_dir.name}/{v.replace('+', '_')}",
                 message=f"{run_dir.name}: {v}", max_mb=95)
    return summary
