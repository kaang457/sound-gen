"""Datasets: download -> preprocess -> one packed int16 file -> training batches.

Every source is downloaded by code. The only exceptions are your own recordings
(data/custom/<label>/...) and Freesound, which needs a free API key
(https://freesound.org/apiv2/apply). Attribution and license are kept per clip
in index.csv.

Pack layout (written to Drive, copied to local disk for training):
    <pack>/audio.bin   int16 mono 22.05 kHz, all clips concatenated
    <pack>/index.csv   id,label,source,src_id,offset,length,split,license,attribution
    <pack>/meta.json
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm

from .audio import SR, load_audio, trim_silence, rms_db

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "configs" / "categories.yaml"

SOURCES = {
    "dcase": dict(
        url="https://zenodo.org/records/8091972/files/DCASE_2023_Challenge_Task_7_Dataset.tar.gz?download=1",
        file="DCASE_2023_Challenge_Task_7_Dataset.tar.gz", md5="66fba60eb490017ee06e1afef71f81eb",
        license="CC BY 4.0 (BBC-sourced clips: research use only, see DevMeta.csv)",
        attribution="DCASE 2023 Task 7 Foley Sound Synthesis dataset (Choi et al.)"),
    "esc50": dict(
        url="https://github.com/karoldvl/ESC-50/archive/master.zip", file="ESC-50-master.zip", md5=None,
        license="CC BY-NC 3.0", attribution="ESC-50, K. J. Piczak (2015)"),
    "fsd50k": dict(
        base="https://zenodo.org/records/4060432/files/",
        parts=["FSD50K.dev_audio.z01", "FSD50K.dev_audio.z02", "FSD50K.dev_audio.z03",
               "FSD50K.dev_audio.z04", "FSD50K.dev_audio.z05", "FSD50K.dev_audio.zip"],
        gt="FSD50K.ground_truth.zip", license="per clip CC (see FSD50K metadata)",
        attribution="FSD50K, Fonseca et al. (2022)"),
}


def load_categories(path=CATEGORIES):
    return yaml.safe_load(Path(path).read_text())


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ------------------------------------------------------------------ download
def download(url, dest, md5=None, chunk=1 << 22):
    """Resumable HTTP download with optional md5 check."""
    import requests
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        print("already downloaded:", dest.name)
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:      # already complete
            pass
        else:
            r.raise_for_status()
            mode = "ab" if r.status_code == 206 else "wb"
            start = have if mode == "ab" else 0
            total = int(r.headers.get("content-length", 0)) + start
            with open(part, mode) as f, tqdm(total=total or None, initial=start, unit="B", unit_scale=True,
                                             unit_divisor=1024, desc=dest.name, mininterval=1) as bar:
                for block in r.iter_content(chunk):
                    f.write(block)
                    bar.update(len(block))
    part.rename(dest)
    if md5 and _md5(dest) != md5:
        raise IOError(f"md5 mismatch for {dest.name}; delete it and retry")
    print("downloaded:", dest.name)
    return dest


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ sources
def fetch_dcase(raw):
    s = SOURCES["dcase"]
    archive = download(s["url"], Path(raw) / s["file"], s["md5"])
    out = Path(raw) / "dcase"
    if not out.exists():
        with tarfile.open(archive) as t:
            t.extractall(out)
    return out


def fetch_esc50(raw):
    s = SOURCES["esc50"]
    archive = download(s["url"], Path(raw) / s["file"])
    out = Path(raw) / "esc50"
    if not out.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    return out


def fetch_fsd50k(raw, wanted_labels, exclude):
    """~18 GB download. Extracts only clips carrying a wanted label."""
    s = SOURCES["fsd50k"]
    raw = Path(raw) / "fsd50k"
    out = raw / "audio"
    gt = download(s["base"] + s["gt"] + "?download=1", raw / s["gt"])
    with zipfile.ZipFile(gt) as z:
        dev_csv = [n for n in z.namelist() if n.endswith("dev.csv")][0]
        rows = list(csv.DictReader(z.read(dev_csv).decode().splitlines()))
    keep = {}
    for r in rows:
        labels = set(r["labels"].split(","))
        if labels & set(exclude):
            continue
        hit = labels & set(wanted_labels)
        if hit:
            keep[r["fname"]] = sorted(hit)
    print(f"FSD50K: {len(keep)} clips match the wanted labels")
    if out.exists() and len(list(out.glob("*.wav"))) >= len(keep) * 0.99:
        return out, keep
    for p in s["parts"]:
        download(s["base"] + p + "?download=1", raw / p)
    merged = raw / "dev_unsplit.zip"
    if not merged.exists():
        if not shutil.which("zip"):
            subprocess.run(["apt-get", "install", "-y", "-qq", "zip"], check=True)
        subprocess.run(["zip", "-q", "-s", "0", str(raw / "FSD50K.dev_audio.zip"), "--out", str(merged)], check=True)
        for p in s["parts"]:                       # free ~18 GB
            (raw / p).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    print("FSD50K: merging the split archive (a few minutes)...", flush=True)
    with zipfile.ZipFile(merged) as z:
        names = [n for n in z.namelist() if n.endswith(".wav") and Path(n).stem in keep]
        for name in tqdm(names, desc="FSD50K extract", unit="clip"):
            (out / f"{Path(name).stem}.wav").write_bytes(z.read(name))
    merged.unlink()
    return out, keep


def fetch_freesound(raw, api_key, label, queries, per_query=100,
                    licenses=('"Creative Commons 0"', '"Attribution"')):
    """Downloads HQ mp3 previews (token auth, no OAuth needed)."""
    import requests
    out = Path(raw) / "freesound" / label.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for q in queries:
        url = "https://freesound.org/apiv2/search/text/"
        params = dict(query=q, page_size=min(per_query, 150), token=api_key,
                      fields="id,name,previews,license,username,duration,tags",
                      filter=f"duration:[0.3 TO 30] license:({' OR '.join(licenses)})")
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        for s in tqdm(r.json().get("results", []), desc=f"Freesound {label}: {q}", unit="file"):
            if str(s["id"]) in meta:
                continue
            f = out / f"{s['id']}.mp3"
            audio = requests.get(s["previews"]["preview-hq-mp3"], timeout=60)
            if audio.ok:
                f.write_bytes(audio.content)
                meta[str(s["id"])] = dict(name=s["name"], user=s["username"], license=s["license"],
                                          url=f"https://freesound.org/s/{s['id']}/")
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"Freesound {label}: {len(meta)} files")
    return out, meta


# ------------------------------------------------------------------ collect
def collect(raw, sources=("dcase", "esc50"), freesound_key=None, custom_dir=None, cats=None,
            only_labels=None):
    """-> list of dict(path, label, source, src_id, license, attribution)."""
    cats = cats or load_categories()
    table = cats["categories"]
    if only_labels:
        table = {k: v for k, v in table.items() if k in only_labels}
    items = []

    def lookup(source):
        m = {}
        for label, spec in table.items():
            for name in spec.get(source, []) or []:
                m[_norm(name)] = label
        return m

    if "dcase" in sources:
        root = fetch_dcase(raw)
        m = lookup("dcase")
        bbc = _dcase_bbc_files(root)
        for wav in root.rglob("*.wav"):
            label = next((m[_norm(p)] for p in wav.parts[::-1] if _norm(p) in m), None)
            if label:
                lic = SOURCES["dcase"]["license"] if wav.name in bbc else "CC BY 4.0"
                items.append(dict(path=str(wav), label=label, source="dcase", src_id=wav.stem,
                                  license=lic, attribution=SOURCES["dcase"]["attribution"],
                                  public_ok=wav.name not in bbc))
    if "esc50" in sources:
        root = fetch_esc50(raw)
        m = lookup("esc50")
        meta = next(root.rglob("esc50.csv"))
        audio_dir = meta.parent.parent / "audio"
        for r in csv.DictReader(open(meta)):
            label = m.get(_norm(r["category"]))
            if label:
                items.append(dict(path=str(audio_dir / r["filename"]), label=label, source="esc50",
                                  src_id=r["src_file"], license=SOURCES["esc50"]["license"],
                                  attribution=SOURCES["esc50"]["attribution"], public_ok=False))
    if "fsd50k" in sources:
        m = lookup("fsd50k")
        wanted = [n for spec in table.values() for n in spec.get("fsd50k", []) or []]
        root, keep = fetch_fsd50k(raw, wanted, cats.get("fsd50k_exclude", []))
        for fname, labels in keep.items():
            label = next((m[_norm(n)] for n in labels if _norm(n) in m), None)
            if label:
                items.append(dict(path=str(root / f"{fname}.wav"), label=label, source="fsd50k",
                                  src_id=fname, license=SOURCES["fsd50k"]["license"],
                                  attribution=f"FSD50K clip {fname} (freesound.org/s/{fname})",
                                  public_ok=False))
    if "freesound" in sources:
        if not freesound_key:
            print("Freesound skipped: no API key (https://freesound.org/apiv2/apply)")
        else:
            for label, spec in table.items():
                if spec.get("freesound"):
                    root, meta = fetch_freesound(raw, freesound_key, label, spec["freesound"])
                    for fid, info in meta.items():
                        items.append(dict(path=str(root / f"{fid}.mp3"), label=label, source="freesound",
                                          src_id=fid, license=info["license"],
                                          attribution=f"{info['name']} by {info['user']} ({info['url']})",
                                          public_ok="noncommercial" not in info["license"].lower()))
    if custom_dir and Path(custom_dir).exists():
        root = Path(custom_dir)
        for f in root.rglob("*"):
            if f.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"):
                label = "/".join(f.relative_to(root).parts[:-1])
                if label:
                    items.append(dict(path=str(f), label=label, source="custom", src_id=f.stem,
                                      license="user", attribution="user recording", public_ok=False))
    if only_labels:
        items = [i for i in items if i["label"] in only_labels]
    print("collected:", dict(Counter(i["label"] for i in items)))
    return items


def _dcase_bbc_files(root):
    names = set()
    for meta in root.rglob("*Meta*.csv"):
        for line in meta.read_text(errors="ignore").splitlines():
            if "bbc" in line.lower():
                names.update(re.findall(r"[\w\-.]+\.wav", line))
    return names


# ------------------------------------------------------------------ preprocess + pack
def split_of(src_id, val=0.05, test=0.05):
    """Deterministic split by source recording: chunks of one file never cross splits."""
    u = int(hashlib.sha1(str(src_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "val" if u < test + val else "train"


def build_pack(items, pack_dir, max_chunk_s=6.0, min_rms_db=-55.0, max_per_label=2000, seed=0):
    """Load, resample, trim, chunk and append everything into one int16 file."""
    if not items:
        raise ValueError("no audio files found for the requested labels: put files in "
                         "MyDrive/sound-gen/custom/<label>/ (e.g. custom/footsteps/snow/*.wav) "
                         "or set a FREESOUND_API_KEY secret")
    pack = Path(pack_dir)
    pack.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it["label"]].append(it)
    rows, offset = [], 0
    bar = tqdm(total=sum(min(len(g), max_per_label) for g in by_label.values()), desc="packing", unit="file")
    with open(pack / "audio.bin", "wb") as bin_f:
        for label, group in sorted(by_label.items()):
            rng.shuffle(group)
            n_label = 0
            bar.set_postfix_str(label)
            n_files = 0
            for it in group:
                if n_label >= max_per_label:
                    break
                n_files += 1
                bar.update(1)
                try:
                    x = load_audio(it["path"], SR)
                except Exception as e:
                    tqdm.write(f"skip (unreadable): {it['path']} {e}")
                    continue
                peak = np.abs(x).max() if len(x) else 0
                if peak < 1e-4:
                    continue
                x = trim_silence(x / peak * 0.9)
                L = int(max_chunk_s * SR)
                for k, start in enumerate(range(0, max(1, len(x)), L)):
                    chunk = x[start: start + L]
                    if len(chunk) < int(0.25 * SR) or rms_db(chunk) < min_rms_db:
                        continue
                    pcm = np.round(np.clip(chunk, -1, 1) * 32767).astype("<i2")
                    bin_f.write(pcm.tobytes())
                    rows.append(dict(id=f"{it['source']}_{it['src_id']}_{k}", label=label,
                                     source=it["source"], src_id=it["src_id"], offset=offset,
                                     length=len(pcm), split=split_of(f"{it['source']}_{it['src_id']}"),
                                     license=it["license"], attribution=it["attribution"],
                                     public_ok=int(bool(it.get("public_ok")))))
                    offset += len(pcm)
                    n_label += 1
            bar.update(max(0, min(len(group), max_per_label) - n_files))
            tqdm.write(f"{label}: {n_label} chunks")
    bar.close()
    with open(pack / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = Counter(r["label"] for r in rows if r["split"] == "train")
    meta = dict(sample_rate=SR, n_clips=len(rows), hours=offset / SR / 3600, train_counts=counts)
    (pack / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"pack: {len(rows)} chunks, {meta['hours']:.2f} h -> {pack}")
    return meta


def read_index(pack_dir):
    with open(Path(pack_dir) / "index.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["offset"], r["length"] = int(r["offset"]), int(r["length"])
        r["pack"] = str(pack_dir)
    return rows


# ------------------------------------------------------------------ training data
class PackDataset:
    """Random crops from one or more packs, with event-aware cropping and augmentation."""

    def __init__(self, packs, vocab, split="train", segment=16384, labels=None,
                 augment=True, min_clips=1):
        self.segment, self.augment = segment, augment
        self.vocab = {lab: i for i, lab in enumerate(vocab)}
        self.audio = {str(p): np.memmap(Path(p) / "audio.bin", dtype="<i2", mode="r") for p in packs}
        rows = [r for p in packs for r in read_index(p) if r["split"] == split]
        counts = Counter(r["label"] for r in rows)
        allowed = set(labels) if labels else set(vocab)
        self.rows = [r for r in rows if r["label"] in allowed and r["label"] in self.vocab
                     and counts[r["label"]] >= min_clips]
        self.counts = Counter(r["label"] for r in self.rows)

    def __len__(self):
        return len(self.rows)

    def sample_weights(self, alpha=0.5):
        """Class balancing: weight ~ count^-alpha (alpha=1 fully balanced, 0 natural)."""
        return np.array([self.counts[r["label"]] ** -alpha for r in self.rows], dtype=np.float64)

    def __getitem__(self, i):
        r = self.rows[i]
        a = self.audio[r["pack"]][r["offset"]: r["offset"] + r["length"]]
        n = self.segment
        if len(a) <= n:
            x = np.zeros(n, np.float32)
            start = np.random.randint(0, n - len(a) + 1)
            x[start: start + len(a)] = a / 32768.0
        else:
            # Event-aware crop: mostly the loudest of 4 candidates, sometimes random
            # (so decays/tails are learned too).
            starts = np.random.randint(0, len(a) - n + 1, size=4)
            if np.random.rand() < 0.7:
                energy = [np.mean(np.square(a[s: s + n: 8].astype(np.float32))) for s in starts]
                s = starts[int(np.argmax(energy))]
            else:
                s = starts[0]
            x = a[s: s + n].astype(np.float32) / 32768.0
        if self.augment:
            x = x * (10 ** (np.random.uniform(-6.0, 3.0) / 20))
            if np.random.rand() < 0.5:
                x = -x
            x = np.clip(x, -1.0, 1.0)
        return x.astype(np.float32), self.vocab[r["label"]]
