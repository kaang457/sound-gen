"""Openly licensed audio that may be used to train a commercial model: download, filter by licence, caption.

Every kept file is written as   <out>/<source>/<id>.flac  +  <id>.txt (caption)   and one row in <out>/manifest.csv
(source, id, path, seconds, licence, licence_url, author, attribution, url, title, caption). The folder is directly a
Stable Audio 3 LoRA dataset (audio + matching .txt), and manifest.csv is the attribution list a product needs.

Kept licences (commercial use allowed, no share-alike): CC0, public domain, CC-BY 3.0/4.0, OGA-BY.
Dropped: any NC (non-commercial), ND, SA (share-alike: unclear for model weights), Sampling+, GPL, unknown.
Not a legal opinion: check the licence texts before a product ships.

Sources (all run in Colab; each has a size / count budget):
    freesound_laion   ~455 k Freesound clips with per-clip licence flags (HF parquet, 679 GB in total: streamed
                      shard by shard, only kept clips are stored)
    clotho            ~7 k Freesound clips with 5 human captions each (Zenodo); per-file licence in its metadata
    fsd50k            ~51 k human-labelled Freesound clips; only the CC0 / CC-BY ones (per-clip licence)
    fsdnoisy18k       18.5 k Freesound clips (42.5 h, 20 classes), every clip CC0 or CC-BY
    opengameart       game sound-effect packs (CC0 / CC-BY / OGA-BY), item pages scraped politely
    kenney            Kenney.nl audio packs, all CC0
    wikimedia         Wikimedia Commons audio categories (sound effects, animals, vehicles, weapons, nature ...)
    inaturalist       animal sounds from iNaturalist observations with CC0 / CC-BY sound licences (API)
    blender_films     Blender open movies (CC-BY): video + final mix, for testing / evaluating video-to-audio
"""
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "sound-gen-dataset/0.1 (research; contact: repo kaang457/sound-gen)"
SR = 44100
FIELDS = ["source", "id", "path", "seconds", "license", "license_url", "author", "attribution", "url", "title",
          "caption"]


# ------------------------------------------------------------------------------------------------ licences
def licence(text_or_url):
    """Normalise a licence string / URL. -> (short name, kept?)"""
    s = str(text_or_url or "").strip().lower()
    if not s:
        return "unknown", False
    if "publicdomain/zero" in s or s in ("cc0", "cc0-1.0", "cc0 1.0") or "cc0" in s:
        return "CC0", True
    if "publicdomain/mark" in s or "public domain" in s or s in ("pd", "public-domain") or s.startswith("pd-"):
        return "PD", True
    if "sampling" in s:
        return "Sampling+", False
    if re.search(r"\bnc\b|noncommercial|non-commercial|-nc|nc-", s):
        return "NC", False
    if re.search(r"\bnd\b|noderiv|-nd", s):
        return "ND", False
    if re.search(r"\bsa\b|sharealike|share-alike|-sa", s):
        return "SA", False
    if "gpl" in s:
        return "GPL", False
    if "oga-by" in s or "oga by" in s:
        return "OGA-BY", True
    if re.search(r"licenses/by/|cc[- ]by|^by[- ]?\d|attribution", s):
        return "CC-BY", True
    return "unknown", False


# ------------------------------------------------------------------------------------------------ store
class Store:
    """Output folder + manifest; resumable (ids already in the manifest are skipped)."""

    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = self.out / "manifest.csv"
        self.seen = set()
        self.bytes = {}
        if self.manifest.exists():
            with open(self.manifest, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.seen.add((r["source"], r["id"]))
                    p = self.out / r["path"]
                    self.bytes[r["source"]] = self.bytes.get(r["source"], 0) + (p.stat().st_size if p.exists() else 0)

    def has(self, source, id_):
        return (source, str(id_)) in self.seen

    def gb(self, source):
        return self.bytes.get(source, 0) / 2**30

    def count(self, source):
        return sum(1 for s, _ in self.seen if s == source)

    def add(self, source, id_, src_file_or_bytes, caption, lic, lic_url="", author="", attribution="", url="",
            title="", max_seconds=30.0, ext_hint=""):
        """Convert to 44.1 kHz FLAC (stereo kept, <= max_seconds), write caption + manifest row. -> path or None"""
        id_ = str(id_)
        if self.has(source, id_):
            return None
        d = self.out / source
        d.mkdir(exist_ok=True)
        stem = _safe(id_)
        dst = d / f"{stem}.flac"
        tmp = None
        if isinstance(src_file_or_bytes, (bytes, bytearray)):
            tmp = d / f".{stem}.in{ext_hint or ''}"
            tmp.write_bytes(src_file_or_bytes)
            src = tmp
        else:
            src = Path(src_file_or_bytes)
        try:
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-t", str(max_seconds),
                                "-ar", str(SR), "-sample_fmt", "s16", "-c:a", "flac", str(dst)], capture_output=True)
            if r.returncode or not dst.exists() or dst.stat().st_size < 2000:
                dst.unlink(missing_ok=True)
                return None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        secs = _duration(dst)
        if secs is None or secs < 0.2:
            dst.unlink(missing_ok=True)
            return None
        caption = _clean_caption(caption or title)
        dst.with_suffix(".txt").write_text(caption, encoding="utf-8")
        new = not self.manifest.exists()
        with open(self.manifest, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(dict(source=source, id=id_, path=str(dst.relative_to(self.out)), seconds=f"{secs:.2f}",
                            license=lic, license_url=lic_url, author=author,
                            attribution=attribution or f"{title} by {author} ({lic})", url=url, title=title,
                            caption=caption))
        self.seen.add((source, id_))
        self.bytes[source] = self.bytes.get(source, 0) + dst.stat().st_size
        return dst

    def summary(self):
        rows = list(csv.DictReader(open(self.manifest, newline="", encoding="utf-8"))) if self.manifest.exists() else []
        out = {}
        for r in rows:
            k = (r["source"], r["license"])
            n, s = out.get(k, (0, 0.0))
            out[k] = (n + 1, s + float(r["seconds"] or 0))
        return {f"{a} / {b}": dict(files=n, hours=round(s / 3600, 2)) for (a, b), (n, s) in sorted(out.items())}


def _safe(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("._")
    return s[:80] if len(s) <= 80 else s[:60] + "_" + hashlib.md5(s.encode()).hexdigest()[:10]


def _duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def _clean_caption(s, max_len=300):
    s = re.sub(r"<[^>]+>", " ", str(s or ""))
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;-")
    return s[:max_len]


def _get(url, tries=5, timeout=60, binary=False, headers=None):
    h = {"User-Agent": UA, **(headers or {})}
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except Exception as e:                                         # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
            last = e  # noqa: F841


def _json(url, **kw):
    return json.loads(_get(url, **kw))


def _download(url, dest):
    from .data import download
    return download(url, dest)


# ------------------------------------------------------------------------------------------------ freesound (HF)
FREESOUND_HF = "benjamin-paine/freesound-laion-640k"
GAME_WORDS = ("impact", "hit", "punch", "explosion", "gun", "shot", "weapon", "sword", "metal", "glass", "footstep",
              "steps", "door", "creak", "whoosh", "swoosh", "laser", "sci-fi", "scifi", "spaceship", "engine", "motor",
              "car", "horse", "animal", "dog", "bird", "monster", "growl", "roar", "creature", "zombie", "magic",
              "spell", "ui", "click", "button", "coin", "pickup", "power", "alarm", "rain", "wind", "thunder", "water",
              "fire", "forest", "cave", "city", "crowd", "ambience", "ambient", "atmosphere", "foley", "cloth",
              "wood", "stone", "debris", "crash", "break", "bell", "horn", "ship", "sea", "wave", "boat", "machine",
              "mechanical", "robot", "servo", "drone", "insect", "jump", "land", "swing", "bow", "arrow", "shield")


def freesound_laion(store, budget_gb=40.0, max_seconds=30.0, game_only=True, max_shards=None, keep_shards=False):
    """Stream the HF parquet shards; keep clips whose licence is CC0 / CC-BY (per-clip field) and, with game_only,
    whose title/tags/description mention a game-relevant word. Shards are deleted after use."""
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(FREESOUND_HF, repo_type="dataset") if f.endswith(".parquet"))
    print(f"freesound_laion: {len(files)} parça (shard); bütçe {budget_gb} GB, şu an {store.gb('freesound'):.1f} GB")
    for i, f in enumerate(files[: max_shards or len(files)]):
        if store.gb("freesound") >= budget_gb:
            break
        p = hf_hub_download(FREESOUND_HF, f, repo_type="dataset")
        kept = 0
        try:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    fid = r.get("freesound_id")
                    if fid is None or store.has("freesound", fid):
                        continue
                    lic_raw = r.get("license")
                    lic_name, ok = licence(_hf_label(pf, "license", lic_raw))
                    if not ok or r.get("commercial_use") is False:
                        continue
                    tags = r.get("tags") or []
                    text = " ".join([str(r.get("title") or ""), " ".join(map(str, tags)), str(r.get("description") or "")])
                    if game_only and not any(w in text.lower() for w in GAME_WORDS):
                        continue
                    a = r.get("audio") or {}
                    data = a.get("bytes") if isinstance(a, dict) else a
                    if not data:
                        continue
                    ext = Path(str(a.get("path") or "x.wav")).suffix if isinstance(a, dict) else ".wav"
                    cap = _fs_caption(r.get("title"), tags, r.get("description"))
                    if store.add("freesound", fid, data, cap, lic_name, author=r.get("username") or "",
                                 url=f"https://freesound.org/s/{fid}/", title=r.get("title") or "",
                                 max_seconds=max_seconds, ext_hint=ext):
                        kept += 1
                    if store.gb("freesound") >= budget_gb:
                        break
                if store.gb("freesound") >= budget_gb:
                    break
        finally:
            if not keep_shards:
                Path(p).unlink(missing_ok=True)
        print(f"  {i + 1}/{len(files)} {f}: +{kept} · toplam {store.count('freesound')} dosya, {store.gb('freesound'):.1f} GB")


def _hf_label(pf, col, v):
    """ClassLabel ints -> names (the dataset stores the licence as a class label)."""
    if isinstance(v, str) or v is None:
        return v
    try:
        meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
        names = meta["info"]["features"][col]["names"]
        return names[int(v)]
    except Exception:                                                  # noqa: BLE001
        return ["CC0-1.0", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-NC-3.0", "CC-BY-NC-4.0", "Sampling+"][int(v)] \
            if isinstance(v, int) and 0 <= v < 6 else str(v)


def _fs_caption(title, tags, desc):
    t = re.sub(r"\.(wav|mp3|flac|ogg|aiff?)$", "", str(title or ""), flags=re.I).replace("_", " ").replace("-", " ")
    tg = ", ".join(str(x) for x in (tags or [])[:8])
    d = _clean_caption(desc, 160)
    return ". ".join(x for x in (t.strip(), tg, d) if x)


# ------------------------------------------------------------------------------------------------ clotho
CLOTHO = "https://zenodo.org/records/4743815/files/"
CLOTHO_SPLITS = {"development": "development", "validation": "validation", "evaluation": "evalution"}


def clotho(store, raw, splits=("development", "validation", "evaluation")):
    """Clotho v2: human captions; keep files whose Freesound licence (metadata csv) is CC0 / CC-BY."""
    raw = Path(raw) / "clotho"
    raw.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "py7zr"], check=True)
    import py7zr
    for split in splits:
        meta = _download(CLOTHO + f"clotho_metadata_{split}.csv?download=1", raw / f"meta_{split}.csv")
        caps = _download(CLOTHO + f"clotho_captions_{split}.csv?download=1", raw / f"caps_{split}.csv")
        arch = raw / f"audio_{split}.7z"
        d = raw / split
        if not d.exists():
            _download(CLOTHO + f"clotho_audio_{CLOTHO_SPLITS[split]}.7z?download=1", arch)
            with py7zr.SevenZipFile(arch) as z:
                z.extractall(raw)
            arch.unlink(missing_ok=True)
        captions = {r["file_name"]: r for r in csv.DictReader(open(caps, encoding="utf-8", errors="replace"))}
        kept = 0
        for r in csv.DictReader(open(meta, encoding="utf-8", errors="replace")):
            name, ok = licence(r.get("license"))
            if not ok:
                continue
            fn = r["file_name"]
            src = next(iter(raw.rglob(fn)), None)
            if src is None:
                continue
            c = captions.get(fn, {})
            cap = (c.get("caption_1") or r.get("keywords") or fn).strip()
            sid = r.get("sound_id") or fn
            if store.add("clotho", sid, src, cap, name, lic_url=r.get("license", ""),
                         author=r.get("manufacturer", ""), url=r.get("sound_link", ""), title=fn):
                kept += 1
        print(f"clotho {split}: +{kept}")


# ------------------------------------------------------------------------------------------------ fsd50k / fsdnoisy18k
def _label_words(labels):
    return ", ".join(l.replace("_and_", " and ").replace("_", " ") for l in labels)


def fsd50k(store, raw, splits=("dev", "eval")):
    """FSD50K (~51 k human-labelled Freesound clips): only the CC0 / CC-BY clips (per-clip licence in its metadata).
    ~24 GB download for both splits."""
    from .data import fetch_fsd50k
    for split in splits:
        out, keep, clips = fetch_fsd50k(raw, [], exclude=[], commercial=True, keep_all=True, split=split)
        kept = 0
        for fname, labels in keep.items():
            src = Path(out) / f"{fname}.wav"
            c = clips.get(str(fname), {})
            name, ok = licence(c.get("license"))
            if not ok or not src.exists():
                continue
            title = c.get("title", "")
            cap = f"{_label_words(labels)}. {re.sub(r'[_-]+', ' ', Path(title).stem)}"
            if store.add("fsd50k", fname, src, cap, name, c.get("license", ""), author=c.get("uploader", ""),
                         url=f"https://freesound.org/s/{fname}/", title=title):
                kept += 1
        print(f"fsd50k {split}: +{kept}")


FSDNOISY = "https://zenodo.org/records/2529934/files/"


def fsdnoisy18k(store, raw):
    """FSDnoisy18k: 18.5 k Freesound clips, 42.5 h, 20 classes; every clip CC0 or CC-BY (dataset CC-BY 4.0)."""
    import zipfile
    raw = Path(raw) / "fsdnoisy18k"
    raw.mkdir(parents=True, exist_ok=True)
    files = {k: _download(FSDNOISY + f"FSDnoisy18k.{k}.zip?download=1", raw / f"{k}.zip")
             for k in ("meta", "doc", "audio_test", "audio_train")}
    labels, lic = {}, {}
    with zipfile.ZipFile(files["meta"]) as z:
        for n in z.namelist():
            if n.endswith(".csv"):
                for r in csv.DictReader(z.read(n).decode("utf-8", "replace").splitlines()):
                    if r.get("fname"):
                        labels[r["fname"]] = r.get("label", "")
    with zipfile.ZipFile(files["doc"]) as z:
        for n in z.namelist():
            if "licen" in n.lower() and n.endswith(".csv"):
                rows = list(csv.DictReader(z.read(n).decode("utf-8", "replace").splitlines()))
                for r in rows:
                    key = r.get("fname") or r.get("filename") or next(iter(r.values()), "")
                    lic[key] = next((v for k, v in r.items() if "licen" in k.lower()), "")
    kept = 0
    for k in ("audio_test", "audio_train"):
        with zipfile.ZipFile(files[k]) as z:
            for n in z.namelist():
                if not n.endswith(".wav"):
                    continue
                fn = Path(n).name
                l_raw = lic.get(fn) or lic.get(Path(fn).stem) or "CC0 or CC-BY (FSDnoisy18k: every clip)"
                name, ok = licence(l_raw)
                if not ok and "every clip" in l_raw:
                    name, ok = "CC0/CC-BY", True
                if not ok or store.has("fsdnoisy18k", fn):
                    continue
                cap = _label_words([labels.get(fn, "")]) or "sound"
                if store.add("fsdnoisy18k", fn, z.read(n), cap, name, l_raw, url="https://zenodo.org/records/2529934",
                             title=fn, ext_hint=".wav"):
                    kept += 1
    print(f"fsdnoisy18k: +{kept}")


# ------------------------------------------------------------------------------------------------ opengameart
OGA = "https://opengameart.org"


def opengameart(store, raw, max_items=400, max_gb=5.0, delay=1.5):
    """Sound-effect items, most downloaded first; licence read from each item page; archives unpacked."""
    raw = Path(raw) / "oga"
    raw.mkdir(parents=True, exist_ok=True)
    done = raw / "items_done.txt"
    seen = set(done.read_text().split()) if done.exists() else set()
    n_items = 0
    for page in range(0, 200):
        if n_items >= max_items or store.gb("opengameart") >= max_gb:
            break
        url = (f"{OGA}/art-search-advanced?keys=&field_art_type_tid%5B%5D=13&sort_by=count&sort_order=DESC"
               f"&items_per_page=48&page={page}")
        html = _get(url)
        items = list(dict.fromkeys(re.findall(r'href="(/content/[a-z0-9\-]+)"', html)))
        items = [i for i in items if i not in ("/content/faq",)]
        if not items:
            break
        for it in items:
            if it in seen or n_items >= max_items or store.gb("opengameart") >= max_gb:
                continue
            time.sleep(delay)
            try:
                n_items += _oga_item(store, raw, it)
            except Exception as e:                                     # noqa: BLE001
                print(f"  oga {it}: {type(e).__name__}: {e}")
            seen.add(it)
            done.write_text("\n".join(sorted(seen)))
        print(f"opengameart sayfa {page}: {n_items} öğe, {store.count('opengameart')} dosya, "
              f"{store.gb('opengameart'):.2f} GB")


def oga_parse(html):
    """-> dict(title, licences [(name, ok, url)], files [url], tags [str], author)"""
    title = re.search(r"<title>(.*?)\|", html, re.S)
    lic_block = re.search(r"License\(s\):(.*?)(?:Collections:|File\(s\):|<div class=\"field field-name-field-art-files)",
                          html, re.S)
    lic_urls = re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', lic_block.group(1)) if lic_block else []
    lics = [(licence(u + " " + t)[0], licence(u + " " + t)[1], u) for u, t in lic_urls]
    files = list(dict.fromkeys(re.findall(r'href="(https?://opengameart\.org/sites/default/files/[^"]+)"', html)))
    tags = re.findall(r'href="/art-search\?keys=([^"]+)"', html)
    author = re.search(r'Author:(?:(?!</a>).){0,400}?<a [^>]*>([^<]+)</a>', html, re.S)
    return dict(title=(title.group(1).strip() if title else ""), licences=lics, files=files,
                tags=[urllib.parse.unquote_plus(t) for t in tags], author=author.group(1).strip() if author else "")


AUDIO_EXT = (".wav", ".ogg", ".mp3", ".flac", ".aif", ".aiff", ".m4a", ".opus")


def _oga_item(store, raw, path):
    html = _get(OGA + path)
    info = oga_parse(html)
    ok = [l for l in info["licences"] if l[1]]
    if not ok:                         # an item may be multi-licensed: any kept licence is enough
        return 0
    lic_name, _, lic_url = ok[0]
    kept = 0
    for fu in info["files"]:
        fn = urllib.parse.unquote(fu.rsplit("/", 1)[-1])
        ext = Path(fn).suffix.lower()
        if ext not in AUDIO_EXT + (".zip",):
            continue
        local = raw / "files" / _safe(path) / _safe(fn)
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            local.write_bytes(_get(fu, binary=True, timeout=300))
        members = _unpack(local) if ext == ".zip" else [local]
        for m in members:
            words = re.sub(r"[_\-.]+", " ", m.stem)
            cap = f"{words}. {info['title']}. " + ", ".join(info["tags"][:6])
            if store.add("opengameart", f"{path.rsplit('/', 1)[-1]}__{m.stem}", m, cap, lic_name, lic_url,
                         author=info["author"], url=OGA + path, title=f"{info['title']} / {m.name}"):
                kept += 1
    return 1 if kept else 0


def _unpack(zip_path):
    import zipfile
    d = zip_path.with_suffix("")
    if not d.exists():
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(d)
        except zipfile.BadZipFile:
            return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("._"))


# ------------------------------------------------------------------------------------------------ kenney
KENNEY_PACKS = ["impact-sounds", "interface-sounds", "rpg-audio", "sci-fi-sounds", "digital-audio", "casino-audio",
                "ui-audio", "foley-sounds"]


def kenney(store, raw, packs=KENNEY_PACKS):
    """Kenney.nl audio packs (CC0). The zip link is read from each asset page."""
    raw = Path(raw) / "kenney"
    raw.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        try:
            html = _get(f"https://kenney.nl/assets/{pack}")
        except Exception as e:                                         # noqa: BLE001
            print(f"kenney {pack}: sayfa açılmadı ({e})")
            continue
        z = re.findall(r'href="([^"]+\.zip)"', html)
        if not z:
            print(f"kenney {pack}: zip bağlantısı bulunamadı (sayfa yapısı değişmiş olabilir)")
            continue
        url = urllib.parse.urljoin("https://kenney.nl/", z[0])
        local = raw / f"{pack}.zip"
        if not local.exists():
            local.write_bytes(_get(url, binary=True, timeout=300))
        kept = 0
        for m in _unpack(local):
            words = re.sub(r"([a-z])([A-Z])", r"\1 \2", m.stem)
            words = re.sub(r"[_\-]+|\d+", " ", words).strip().lower()
            cap = f"{words}, video game sound effect ({pack.replace('-', ' ')})"
            if store.add("kenney", f"{pack}__{m.stem}", m, cap, "CC0",
                         "https://creativecommons.org/publicdomain/zero/1.0/", author="Kenney (kenney.nl)",
                         url=f"https://kenney.nl/assets/{pack}", title=f"{pack} / {m.name}"):
                kept += 1
        print(f"kenney {pack}: +{kept}")


# ------------------------------------------------------------------------------------------------ wikimedia
WIKI = "https://commons.wikimedia.org/w/api.php"
WIKI_CATEGORIES = ["Sound effects", "Audio files of animals", "Audio files of birds", "Audio files of vehicles",
                   "Audio files of weapons", "Audio files of firearms", "Audio files of explosions",
                   "Sounds of nature", "Audio files of weather", "Audio files of rain", "Audio files of thunder",
                   "Audio files of water", "Audio files of machines", "Audio files of bells", "Audio files of trains",
                   "Audio files of aircraft", "Audio files of ships", "Audio files of footsteps", "Field recordings"]


def wikimedia(store, categories=WIKI_CATEGORIES, depth=2, max_files=5000, max_seconds=30.0):
    """Commons audio files in the categories (and subcategories up to `depth`); licence from extmetadata."""
    todo, seen_cat, n = [(c, 0) for c in categories], set(), 0
    while todo and n < max_files:
        cat, dep = todo.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        cont = {}
        while n < max_files:
            q = dict(action="query", format="json", generator="categorymembers", gcmtitle=f"Category:{cat}",
                     gcmlimit="200", prop="imageinfo", iiprop="url|extmetadata|mime|size",
                     gcmtype="file|subcat", **cont)
            try:
                j = _json(WIKI + "?" + urllib.parse.urlencode(q))
            except Exception as e:                                     # noqa: BLE001
                print(f"wikimedia {cat}: {e}")
                break
            for page in (j.get("query", {}).get("pages", {}) or {}).values():
                title = page.get("title", "")
                if page.get("ns") == 14 and dep < depth:
                    todo.append((title.split(":", 1)[1], dep + 1))
                    continue
                ii = (page.get("imageinfo") or [{}])[0]
                if not str(ii.get("mime", "")).startswith(("audio/", "application/ogg", "video/webm")):
                    continue
                md = ii.get("extmetadata", {})
                lic_s = (md.get("LicenseShortName", {}).get("value", "") + " " +
                         md.get("LicenseUrl", {}).get("value", ""))
                name, ok = licence(lic_s)
                if not ok or store.has("wikimedia", title):
                    continue
                desc = _clean_caption(md.get("ImageDescription", {}).get("value", ""), 200)
                words = re.sub(r"\.(ogg|oga|wav|flac|mp3|opus|webm)$", "", title.split(":", 1)[-1], flags=re.I)
                cap = f"{words.replace('_', ' ')}. {desc}. {cat}"
                try:
                    data = _get(ii["url"], binary=True, timeout=300)
                except Exception:                                      # noqa: BLE001
                    continue
                if store.add("wikimedia", title, data, cap, name, md.get("LicenseUrl", {}).get("value", ""),
                             author=_clean_caption(md.get("Artist", {}).get("value", ""), 120),
                             url=ii.get("descriptionurl", ""), title=title, max_seconds=max_seconds,
                             ext_hint=Path(ii["url"]).suffix):
                    n += 1
            if "continue" not in j:
                break
            cont = {k: v for k, v in j["continue"].items()}
            time.sleep(0.5)
        print(f"wikimedia {cat}: toplam {store.count('wikimedia')} dosya")


# ------------------------------------------------------------------------------------------------ inaturalist
INAT = "https://api.inaturalist.org/v1/observations"
INAT_TAXA = {"Aves": 3, "Mammalia": 40151, "Amphibia": 20978, "Insecta": 47158, "Reptilia": 26036}


def inaturalist(store, taxa=("Aves", "Mammalia", "Amphibia", "Insecta"), per_taxon=1500, max_seconds=30.0):
    """Research-grade observations with sounds whose SOUND licence is CC0 or CC-BY (API filter + re-check)."""
    for name in taxa:
        tid, got, page = INAT_TAXA[name], 0, 1
        while got < per_taxon and page <= 50:
            q = dict(sounds="true", sound_license="cc0,cc-by", quality_grade="research", taxon_id=tid,
                     per_page=200, page=page, order_by="votes")
            try:
                j = _json(INAT + "?" + urllib.parse.urlencode(q))
            except Exception as e:                                     # noqa: BLE001
                print(f"inaturalist {name}: {e}")
                break
            res = j.get("results", [])
            if not res:
                break
            for o in res:
                tx = o.get("taxon") or {}
                common = tx.get("preferred_common_name") or ""
                sci = tx.get("name") or ""
                for s in o.get("sounds") or []:
                    lic_name, ok = licence(s.get("license_code"))
                    if not ok or not s.get("file_url"):
                        continue
                    sid = f"{o['id']}_{s['id']}"
                    if store.has("inaturalist", sid):
                        continue
                    kind = "song and calls" if name == "Aves" else ("calls" if name != "Insecta" else "buzzing and chirping")
                    cap = f"{common or sci} ({sci}) {kind}, field recording in nature"
                    try:
                        data = _get(s["file_url"], binary=True, timeout=120)
                    except Exception:                                  # noqa: BLE001
                        continue
                    if store.add("inaturalist", sid, data, cap, lic_name, author=s.get("attribution", ""),
                                 attribution=s.get("attribution", ""),
                                 url=f"https://www.inaturalist.org/observations/{o['id']}", title=f"{common} {sci}",
                                 max_seconds=max_seconds, ext_hint=Path(urllib.parse.urlparse(s["file_url"]).path).suffix):
                        got += 1
                    if got >= per_taxon:
                        break
                if got >= per_taxon:
                    break
            page += 1
            time.sleep(1.0)                                            # API etiquette: ~1 request / s
        print(f"inaturalist {name}: +{got}")


# ------------------------------------------------------------------------------------------------ blender films
BLENDER_INDEXES = {
    "big_buck_bunny": "https://download.blender.org/peach/bigbuckbunny_movies/",
    "elephants_dream": "https://download.blender.org/ED/",
    "sintel": "https://download.blender.org/durian/movies/",
    "tears_of_steel": "https://download.blender.org/demo/movies/ToS/",
}


def blender_films(dest, max_mb=600):
    """CC-BY open movies with their final sound mix: test / evaluation clips for 05 (cut 10-30 s pieces, strip the
    sound, regenerate, compare with the original). The mix contains music and dialogue: not SFX training data."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, idx in BLENDER_INDEXES.items():
        try:
            html = _get(idx)
        except Exception as e:                                         # noqa: BLE001
            print(f"{name}: {e}")
            continue
        links = re.findall(r'href="([^"?]+\.(?:mp4|mov|m4v|mkv|avi))"', html, re.I)
        sizes = {}
        for l in links:                                                # prefer small HD versions
            m = re.search(r'href="' + re.escape(l) + r'".*?(\d+(?:\.\d+)?)([KMG])\s*$', html, re.M)
            if m:
                sizes[l] = float(m.group(1)) * {"K": 1e-3, "M": 1, "G": 1e3}[m.group(2)]
        cand = sorted((s, l) for l, s in sizes.items() if s <= max_mb) or [(0, l) for l in links[:1]]
        if not cand:
            print(f"{name}: video bulunamadı")
            continue
        l = cand[-1][1]
        out = dest / f"{name}{Path(l).suffix.lower()}"
        if not out.exists():
            _download(urllib.parse.urljoin(idx, l), out)
        print(f"{name}: {out} (CC-BY, Blender Foundation)")
    (dest / "LICENSE.txt").write_text("Blender open movies, (c) Blender Foundation, CC-BY (www.blender.org). "
                                      "Credit the films and the Blender Foundation when you show results.\n")


# ------------------------------------------------------------------------------------------------ helpers
def lora_dir_ok(out):
    """True when every audio file has its caption (what Stable Audio 3's LoRA trainer expects)."""
    out = Path(out)
    audio = [p for p in out.rglob("*.flac")]
    return bool(audio) and all(p.with_suffix(".txt").exists() for p in audio)


def to_items(out, label_fn=None):
    """manifest -> items for soundgen.data.build_pack (label = source unless label_fn(row) says otherwise)."""
    out = Path(out)
    items = []
    for r in csv.DictReader(open(out / "manifest.csv", newline="", encoding="utf-8")):
        items.append(dict(path=str(out / r["path"]), label=(label_fn(r) if label_fn else r["source"]),
                          source=r["source"], src_id=r["id"], license=r["license"],
                          attribution=r["attribution"], public_ok=True, caption=r["caption"]))
    return items


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30
