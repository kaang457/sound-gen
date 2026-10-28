"""Ready-made generator: Stable Audio 3 Small-SFX (Stability AI, May 2026), no training on our side.

Why: our own data (~48 h, research licences) is far too small to train a generator from scratch; this model was
trained by Stability on ~1.28 M licensed / CC recordings (AudioSparx + Freesound CC0/CC-BY/Sampling+) and its
weights are open.

Licence (check the licence text before a product ships): Stability AI Community License: free, incl. commercial use
and ownership of the outputs, for organisations under USD 1 M annual revenue; above that an Enterprise licence.
The text encoder (google/t5gemma-b-b-ul2) comes with the Gemma Terms of Use. Both Hugging Face repos are gated:
log in, accept the terms on each page, put a read token in the Colab secret HF_TOKEN.

    from soundgen import sa3
    gen = sa3.load(P.drive)                               # cache, install, access check, small-sfx, 44.1 kHz stereo, ~0.4 s per batch on a GPU
    sounds = gen.generate("a single pistol gunshot", seconds=3, seed=1)       # 8 candidates (8 seeds)
    files = sa3.reference_run(gen, run_dir, push=colab.push_to_github)        # the reference set -> results/<RUN>

The same object plugs into the studio: Studio(generator=gen).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SA3_GIT = "https://github.com/Stability-AI/stable-audio-3"
SA3_COMMIT = "77e0398"                              # tested with this commit (11 Oct 2026)
GATED = {"small-sfx": ["stabilityai/stable-audio-3-small-sfx", "google/t5gemma-b-b-ul2"],
         "small-sfx-base": ["stabilityai/stable-audio-3-small-sfx-base", "google/t5gemma-b-b-ul2"]}
PREFIX = "TrackType: SFX, "                         # Stability's prompt guide: "more semantically reasonable SFX"
CACHE = Path.home() / ".cache" / "soundgen"


def load(drive_dir=None, model="small-sfx", offline=False, lora=None, lora_strength=1.0):
    """Everything cell 1 of the notebooks needs: local HF cache (+ Drive backup), install, access check, model.
    Kept here (not in the notebook) so a `git pull` in cell 0 brings fixes without reopening the notebook."""
    from . import hfcache
    hfcache.setup(drive_dir, offline=offline)
    install()
    if not offline:
        check_access(model)
    gen = SA3(model, lora=lora, lora_strength=lora_strength)
    hfcache.backup()
    return gen


def use_cache(backup_dir=None, **kw):          # older notebook cells
    from . import hfcache
    return hfcache.setup(backup_dir, **kw)


def backup_cache(tar=None):                      # older notebook cells
    from . import hfcache
    return hfcache.backup(tar)


def install():
    """Install the stable_audio_3 package without touching Colab's torch / numpy (its pyproject pins torch 2.7.1)."""
    need = []
    try:
        import stable_audio_3  # noqa: F401
    except ImportError:
        need.append(f"git+{SA3_GIT}@{SA3_COMMIT}")
    for mod, pkg in [("einops", "einops"), ("einops_exts", "einops-exts"), ("safetensors", "safetensors"),
                     ("soundfile", "soundfile"), ("torchaudio", "torchaudio")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *need], check=True)
    # the T5Gemma text encoder needs a recent transformers (SA3 asks for >= 5.8)
    from importlib.metadata import version
    from packaging.version import Version
    tv = Version(version("transformers"))
    if tv < Version("5.8"):
        loaded = "transformers" in sys.modules
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.8"], check=True)
        if loaded:
            raise RuntimeError(f"transformers {tv} -> upgraded; Runtime → Restart session, then run the cells again")
    return need


def install_train():
    """Extra packages for Stable Audio 3's LoRA trainer + the trainer script itself (not part of the pip package)."""
    install()
    need = []
    for mod, pkg in [("pytorch_lightning", "pytorch_lightning==2.5.5"), ("dill", "dill"), ("wandb", "wandb")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
    # torchaudio >= 2.9 reads audio through torchcodec
    try:
        from importlib.metadata import version
        from packaging.version import Version
        if Version(version("torchaudio").split("+")[0]) >= Version("2.9"):
            try:
                import torchcodec  # noqa: F401
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchcodec"], check=True)
    except Exception:                                                  # noqa: BLE001
        pass
    repo = CACHE / "stable-audio-3"
    if not (repo / "scripts" / "train_lora.py").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", SA3_GIT, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", SA3_COMMIT], check=True)
    return repo / "scripts" / "train_lora.py"


def check_access(model="small-sfx"):
    """Fail early, with the page to open, if the token cannot read a gated repo."""
    import os
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    tok = os.environ.get("HF_TOKEN")
    bad = []
    for repo in GATED[model]:
        try:
            auth_check(repo, token=tok)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            bad.append(f"  https://huggingface.co/{repo}  ({type(e).__name__})")
    if bad:
        raise RuntimeError(("HF_TOKEN yok. " if not tok else "") + "Bu sayfalarda giriş yapıp lisansı kabul et, "
                           "sonra Colab 🔑 Secrets → HF_TOKEN (read token):\n" + "\n".join(bad))
    print("Hugging Face erişimi tamam:", ", ".join(GATED[model]))


class SA3:
    """Stable Audio 3 behind the studio's generator interface: generate(...) -> list of sound dicts."""
    kind = "Stable Audio 3"

    def __init__(self, model="small-sfx", device=None, steps=None, prefix=PREFIX, lora=None, lora_strength=1.0):
        import torch
        from stable_audio_3 import StableAudioModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = model
        from . import hfcache
        self.m = hfcache.load(lambda: StableAudioModel.from_pretrained(model, device=self.device), GATED[model])
        self.sr = int(self.m.model.sample_rate)
        self.base = model.endswith("-base")
        self.steps = steps or (50 if self.base else 8)        # post-trained: 8 (more does not help); base: ~50
        self.prefix = prefix
        self.step = None                                       # (our generators report their training step)
        self.lora = None
        if lora:
            # LoRAs are trained on the -base checkpoints (Stable Audio 3's trainer only accepts those)
            paths = [str(x) for x in (lora if isinstance(lora, (list, tuple)) else [lora])]
            self.m.load_lora(paths)
            self.m.set_lora_strength(float(lora_strength))
            self.lora = paths
            self.name = f"{model}+lora"

    def raw(self, prompt, seconds=4.0, n=4, seed=0, cfg=None, negative=None):
        """-> float32 [n, 2, T] stereo in [-1, 1]. cfg / negative only act on the -base checkpoints."""
        kw = dict(prompt=self.prefix + prompt, duration=float(seconds), steps=self.steps, batch_size=n, seed=int(seed))
        if self.base:
            kw.update(cfg_scale=float(cfg or 7.0), negative_prompt=negative)
        x = self.m.generate(**kw)
        return x.float().cpu().numpy()

    def generate(self, prompt, prompt_b="", mix_b=0.0, seconds=4.0, seed=0, cfg=None, n=8):
        """Studio interface. n candidates = one batch with one seed (each item has its own noise).
        A second prompt has no embedding blend here: it is added to the text ("…, with …")."""
        text = prompt.strip()
        if prompt_b and prompt_b.strip() and mix_b > 0:
            text = f"{text}, {'with a hint of' if mix_b < 0.5 else 'mixed with'} {prompt_b.strip()}"
        x = self.raw(text, seconds=seconds, n=n, seed=seed, cfg=cfg)
        out = []
        for k, st in enumerate(x):
            st = _fade(st, self.sr)
            out.append(dict(name=str(k + 1), kind="varyasyon", audio=st.mean(0).astype(np.float32), stereo=st,
                            sr=self.sr, prompt=prompt, prompt_b=prompt_b, mix_b=mix_b, seed=seed, text=text))
        return out


def _fade(st, sr, ms=10):
    k = min(int(sr * ms / 1000), st.shape[-1] // 4)
    st = st.copy()
    if k > 0:
        st[..., -k:] *= np.linspace(1, 0, k, dtype=st.dtype)
    return st


# --------------------------------------------------------------------------------------------- reference set
# A game-audio reference set: what a sound designer asks for first. Durations are realistic for each sound
# (Stability: "set a realistic duration"). Prompts follow the SA3 SFX guide: source + action + recording character.
REFERENCE = [
    # category, short name, prompt, seconds
    ("Silah", "tek tabanca atışı", "A single pistol gunshot, sharp crack with a short punchy tail, close mic, outdoors", 2.5),
    ("Silah", "çatışma", "A gunfight: rapid bursts of automatic rifle fire and scattered pistol shots from several shooters, outdoor battlefield", 6),
    ("Silah", "pompalı tüfek", "A pump-action shotgun: pump rack followed by one heavy shotgun blast with a boomy tail", 3),
    ("Silah", "lazer silah", "A sci-fi laser gun shot, bright zappy pew with a fast pitch drop and a short electric tail", 2),
    ("Silah", "kılıç çarpışması", "Two metal swords clash, bright ringing steel impact with a short metallic shimmer", 2.5),
    ("Silah", "ok ve yay", "A bow releases an arrow: string twang, fast whoosh, the arrow thuds into wood", 2.5),
    ("Patlama / darbe", "büyük patlama", "A large explosion, deep boom with debris falling and a long rumbling tail", 5),
    ("Patlama / darbe", "uzak patlama", "A distant explosion heard from far away, muffled low boom echoing across a valley", 5),
    ("Patlama / darbe", "yumruk", "A heavy punch impact, meaty thud with a fast cloth whoosh, fighting game", 1.5),
    ("Patlama / darbe", "cam kırılması", "A window pane shatters, glass breaking and shards falling on a hard floor", 3),
    ("Bilim kurgu", "uzay gemisi motoru", "A large spaceship engine idling, deep steady sci-fi hum with soft mechanical whir, loopable", 6),
    ("Bilim kurgu", "uzay gemisi geçişi", "A sci-fi spaceship flies past fast from left to right, roaring jet whoosh with doppler pitch shift", 5),
    ("Bilim kurgu", "enerji kalkanı", "A sci-fi energy shield powering up, rising electric hum with a crisp shimmering lock-in", 3),
    ("Bilim kurgu", "ışınlanma", "A sci-fi teleport effect, swirling sparkly whoosh ending in a soft electric pop", 3),
    ("Yaratık", "zombi hırıltısı", "A zombie growl, wet raspy throaty groan, slow and menacing, close mic", 4),
    ("Yaratık", "canavar kükremesi", "A huge monster roar, deep guttural bellow with a gritty distorted edge", 4),
    ("Yaratık", "ejderha kanatları", "A dragon flapping its huge leathery wings, heavy slow whooshes of air", 4),
    ("Yaratık", "köpek havlaması", "A medium-size dog barking several times, outdoors", 4),
    ("Hareket / foley", "çakılda ayak sesi", "Footsteps walking on gravel, steady pace, crunchy close-mic foley", 5),
    ("Hareket / foley", "ahşapta ayak sesi", "Footsteps walking on a creaky old wooden floor indoors, boots", 5),
    ("Hareket / foley", "at dörtnala", "A horse galloping on a dirt road, rhythmic hoof beats passing by", 5),
    ("Hareket / foley", "kapı gıcırtısı", "An old wooden door slowly creaks open with a long squeaky hinge", 4),
    ("Hareket / foley", "sandık açılışı", "A heavy wooden treasure chest unlocks with a metal latch click and the lid creaks open", 3),
    ("Büyü / arayüz", "ateş topu büyüsü", "A fireball spell: whooshing burst of fire launched forward with a crackling roar", 3),
    ("Büyü / arayüz", "iyileştirme büyüsü", "A healing magic spell, soft glittering chimes rising with a warm shimmer", 3),
    ("Büyü / arayüz", "menü tıklaması", "A clean short user interface button click, soft and modern", 1),
    ("Büyü / arayüz", "altın toplama", "A coin pickup sound for a video game, bright short metallic ching", 1),
    ("Büyü / arayüz", "güçlendirme", "A video game power-up pickup, quick rising synth arpeggio with sparkle", 2),
    ("Ortam", "teneke çatıda yağmur", "Heavy rain falling on a tin roof, steady dense patter, loopable ambience", 8),
    ("Ortam", "orman", "Forest ambience in the morning, birds chirping, light breeze in the leaves", 8),
    ("Ortam", "mağarada rüzgâr", "Wind howling through a dark cave, hollow low drone with distant water drips", 8),
    ("Ortam", "kamp ateşi", "A crackling campfire, wood pops and soft flames, close mic", 6),
]
ONE_VS_MANY = [("tek tabanca atışı", "çatışma")]


def _db(v):
    return float(20 * np.log10(max(float(v), 1e-9)))


def reference_run(gen, run_dir, items=None, n=4, seed=1, clap=True, push=None, drive_dir=None):
    """Every reference item -> n variations. Writes WAVs (24-bit stereo), a page (all variations playable), a
    README.md table for GitHub, measures.json; zip of variation 1 for each sound. -> files in <run_dir>/publish."""
    import shutil
    import zipfile
    import soundfile as sf
    import torch
    from .audio import MelSketch, MEL_V3
    from .checks import count_events
    run_dir = Path(run_dir)
    wav_dir = run_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = list(items or REFERENCE)
    mel = MelSketch(MEL_V3)
    rows = []
    for k, (cat, short, prompt, sec) in enumerate(items):
        t0 = time.time()
        x = gen.raw(prompt, seconds=sec, n=n, seed=seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        vars_ = []
        for j, st in enumerate(x):
            st = _fade(st, gen.sr)
            f = wav_dir / f"{k + 1:02d}_{_slug(short)}_v{j + 1}.wav"
            sf.write(f, st.T, gen.sr, subtype="PCM_24")
            mono = st.mean(0)
            vars_.append(dict(wav=str(f), peak_db=_db(np.abs(st).max()), rms_db=_db(np.sqrt((mono ** 2).mean())),
                              events=count_events(mono, gen.sr)))
        m = mel(torch.from_numpy(x[0].mean(0).astype(np.float32))[None])[0].numpy()
        rows.append(dict(i=k + 1, category=cat, short=short, prompt=prompt, seconds=sec, gen_s=dt, vars=vars_, mel=m))
        print(f"{k + 1:2d} {short:22s} {sec:4.1f} s  {dt:5.2f} s üretim  tepe {vars_[0]['peak_db']:6.1f} dBFS  "
              f"olay {[v['events'] for v in vars_]}")
    res = {}
    if clap:
        try:
            res = _clap(rows)
        except Exception as e:                                          # noqa: BLE001  (download etc.)
            print(f"CLAP ölçümü atlandı: {type(e).__name__}: {e}")
    by = {r["short"]: r for r in rows}
    pairs = [dict(one=a, many=b, events_one=[v["events"] for v in by[a]["vars"]],
                  events_many=[v["events"] for v in by[b]["vars"]]) for a, b in ONE_VS_MANY if a in by and b in by]
    for p in pairs:
        print(f"tek / çok: {p['one']} {p['events_one']} olay | {p['many']} {p['events_many']} olay")
    name = run_dir.name
    info = dict(model=f"stabilityai/stable-audio-3-{gen.name}", commit=SA3_COMMIT, steps=gen.steps, sr=gen.sr,
                variations=n, seed=seed, prefix=gen.prefix, gpu=_gpu(), clap=res, pairs=pairs)
    work = run_dir / "page"
    work.mkdir(exist_ok=True)
    (work / "measures.json").write_text(json.dumps(
        dict(info=info, sounds=[{k: v for k, v in r.items() if k != "mel"} for r in rows]),
        indent=1, ensure_ascii=False, default=float))
    _page(rows, info, work / "index.html", f"{name} · Stable Audio 3 ile oyun sesi referans seti")
    (work / "README.md").write_text(_readme(rows, info, name), encoding="utf-8")
    with zipfile.ZipFile(work / f"{name}_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            z.write(r["vars"][0]["wav"], Path(r["vars"][0]["wav"]).name)
    if drive_dir:                                     # every variation, full quality, on Drive (too big for GitHub)
        d = Path(drive_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(d / f"{name}_hepsi.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for r in rows:
                for v in r["vars"]:
                    z.write(v["wav"], Path(v["wav"]).name)
        print("tüm varyasyonlar:", d / f"{name}_hepsi.zip")
    pub = run_dir / "publish"
    pub.mkdir(exist_ok=True)
    for f in ("index.html", "README.md", "measures.json", f"{name}_v1.zip"):
        shutil.copy2(work / f, pub / f)
    files = sorted(pub.iterdir())
    if push:
        push(files, dest=f"results/{name}", message=f"{name}: Stable Audio 3 game-sound reference set", max_mb=95)
    return files


def _gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                                                  # noqa: BLE001
        return "?"


def _clap(rows):
    """Evaluation CLAP (laion/clap-htsat-unfused, a different model from the generator's text encoder):
    for every variation, rank of its own prompt among all reference prompts (1 = best)."""
    import soundfile as sf
    from .evaluate import Clap
    clap = Clap()
    prompts = [r["prompt"] for r in rows]
    t = clap.text(prompts)
    top1, top3 = [], []
    for i, r in enumerate(rows):
        for v in r["vars"]:
            x, sr = sf.read(v["wav"], dtype="float32")
            a = clap.audio([x.mean(1)], sr)[0]
            sims = t @ a
            rank = int((sims > sims[i]).sum()) + 1
            v.update(clap_rank=rank, clap_sim=float(sims[i]))
            top1.append(rank == 1)
            top3.append(rank <= 3)
    res = dict(candidates=len(prompts), top1=float(np.mean(top1)), top3=float(np.mean(top3)))
    print(f"CLAP: kendi isteği {len(prompts)} istek arasında 1. sırada %{100 * res['top1']:.0f}, "
          f"ilk 3'te %{100 * res['top3']:.0f}")
    return res


def _readme(rows, info, name):
    c = info.get("clap") or {}
    lines = [f"# {name}: Stable Audio 3 ile oyun sesi referans seti", "",
             "Hazır model, eğitim yok: **Stable Audio 3 Small-SFX** (Stability AI). Sesleri dinlemek için "
             "[index.html](index.html) dosyasını indirip tarayıcıda aç (her sesin tüm varyasyonları çalınabilir); "
             f"her sesin 1. varyasyonu WAV olarak [`{name}_v1.zip`](./{name}_v1.zip) içinde.", "",
             "| | |", "|---|---|",
             f"| Model | `{info['model']}` (kod commit `{info['commit']}`) |",
             f"| Ayar | {info['steps']} adım, {info['sr']} Hz stereo, istek başı {info['variations']} varyasyon, tohum {info['seed']}, istek öneki `{info['prefix'].strip()}` |",
             f"| GPU | {info['gpu']} |",
             "| Lisans | Stability AI Community License: yıllık geliri 1 M $ altı kuruluşlara ticari kullanım dahil serbest, çıktılar kullanıcının; üstü kurumsal lisans. Metin kodlayıcı Gemma kullanım şartlarıyla. Ürüne girmeden lisans metni okunmalı. |",
             "| Eğitim verisi (Stability) | ~1,28 M kayıt: AudioSparx (lisanslı) + Freesound CC0 / CC-BY / Sampling+ |"]
    if c:
        lines.append(f"| İstek eşleşmesi (CLAP) | kendi isteği {c['candidates']} istek arasında 1. sırada %{100 * c['top1']:.0f}, ilk 3'te %{100 * c['top3']:.0f} |")
    for p in info.get("pairs") or []:
        lines.append(f"| Tek / çok olay | {p['one']}: {p['events_one']} · {p['many']}: {p['events_many']} (varyasyon başına sayılan olay) |")
    lines += ["", "| # | Kategori | Ses | İstek | Süre | Üretim | Tepe | CLAP sırası (varyasyonlar) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ranks = " ".join(str(v.get("clap_rank", "–")) for v in r["vars"])
        lines.append(f"| {r['i']} | {r['category']} | {r['short']} | {r['prompt']} | {r['seconds']:g} s | "
                     f"{r['gen_s']:.2f} s | {r['vars'][0]['peak_db']:.1f} dBFS | {ranks} |")
    lines += ["", "Üretim süresi: o sesin bütün varyasyonları birlikte (tek GPU çağrısı). CLAP sırası 1 = ses, bütün "
              "istekler arasında en çok kendi isteğine benziyor (ölçüm ayrı bir model, dinlemenin yerini tutmaz). "
              "Tepe: 1. varyasyonun en yüksek seviyesi.", ""]
    return "\n".join(lines)


def _page(rows, info, dest, title):
    import html
    from .checks import _png, _ogg
    c = info.get("clap") or {}
    sections, cur = [], None
    for r in rows:
        if r["category"] != cur:
            cur = r["category"]
            sections.append(f"<tr><th colspan='3'>{html.escape(cur)}</th></tr>")
        players = "".join(
            f"<div class='v'><span class='k'>{j + 1}{(' · CLAP ' + str(v['clap_rank'])) if 'clap_rank' in v else ''}"
            f"</span><audio controls preload='none' src='{_ogg(v['wav'])}'></audio></div>"
            for j, v in enumerate(r["vars"]))
        sections.append(
            f"<tr><td class='lab'>{r['i']}. {html.escape(r['short'])}<span class='k'>“{html.escape(r['prompt'])}”</span>"
            f"<span class='k'>{r['seconds']:g} s · üretim {r['gen_s']:.2f} s · tepe {r['vars'][0]['peak_db']:.1f} dBFS</span>"
            f"</td><td><img alt='' src='{_png(r['mel'])}'></td><td>{players}</td></tr>")
    clap_txt = (f" İstek eşleşmesi (CLAP, ayrı bir model): ses kendi isteğine {c['candidates']} istek arasında "
                f"%{100 * c['top1']:.0f} birinci, %{100 * c['top3']:.0f} ilk üç." if c else "")
    intro = (f"<p class='k'>Hazır model, eğitim yok: <b>{html.escape(info['model'])}</b>, {info['steps']} adım, "
             f"{info['sr']} Hz stereo, her istek için {info['variations']} varyasyon (aynı istek, farklı başlangıç "
             f"gürültüsü). GPU: {html.escape(info['gpu'])}.{clap_txt} Lisans: Stability AI Community License "
             f"(yıllık geliri 1 M $ altına ticari kullanım dahil serbest).</p>")
    css = """:root{color-scheme:light dark;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e7e6e2;--acc:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835;--acc:#f2a33a}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 12px}
table{border-collapse:collapse;width:100%}td{border-top:1px solid var(--rule);padding:8px 6px;vertical-align:top}
th{text-align:left;padding:18px 6px 6px;color:var(--acc);font-size:13px;letter-spacing:1px;text-transform:uppercase}
.lab{font-weight:600;max-width:320px}.k{display:block;font-size:12px;font-weight:400;color:var(--ink2)}
img{width:220px;height:auto;border-radius:4px;display:block}.v{display:inline-block;margin:0 8px 6px 0}
audio{height:32px;width:210px}.wrap{overflow-x:auto}"""
    doc = (f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style>"
           f"</head><body><main><h1>{html.escape(title)}</h1>{intro}<div class='wrap'><table>{''.join(sections)}"
           f"</table></div></main></body></html>")
    Path(dest).write_text(doc, encoding="utf-8")


def _slug(s):
    tr = str.maketrans("çğıöşüÇĞİÖŞÜâ", "cgiosuCGIOSUa")
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s.translate(tr).strip().lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep[:48].strip("_") or "x"
