"""Silent gameplay video -> sound track, with ready-made models only (no training).

    video ─► frames (2 fps, with timestamps) ─► PLANNER: Qwen3-VL (Apache-2.0)
                                                 "which sounds, where, when" -> plan (JSON, editable)
          ─► motion energy (25 fps) ─► event times snapped to the nearest visible hit / step
    plan  ─► GENERATOR: Stable Audio 3 Small-SFX, one call per track
               ambience (sea, rain, wind)   one long stereo clip, looped with crossfades if needed
               continuous (engine, horse)   one clip over [start, end]
               events (gunshot, bell, horn) one short clip per event, aligned on its onset, all different
          ─► MIXER: level, distance (air absorption), pan (soundgen.fx) -> stems + stereo mix
          ─► ffmpeg: video + mix -> mp4 (video stream copied untouched)

Every track can be regenerated alone (new prompt / seed) and the mix re-rendered: regenerate().
Everything runs offline once the weights are on disk (HF cache on Drive).

    from soundgen import v2a, sa3
    gen = sa3.SA3(); planner = v2a.Planner()                    # Qwen/Qwen3-VL-8B-Instruct
    job = v2a.Job("clip.mp4", work_dir)
    job.plan = planner.plan(job)                                  # list of tracks (dicts), editable
    job.render(gen)                                               # generate + mix + mp4 -> job.out_mp4, job.stems
    job.edit(2, gain_db=-6, pan=-0.4); job.remix()                # mixing changes: no model call, ~1 s
    job.regenerate(gen, "Sea waves", prompt="Calm sea waves gently lapping on a rocky shore")   # one track only
    # or the studio: soundgen.v2a_ui.build(gen, planner)
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

KINDS = ("ambience", "continuous", "events")
EVENT_GEN_S = 4.0                                      # shortest clip asked from the model for one event
LEVEL_DB = {"low": -18.0, "medium": -10.0, "high": -4.0}
DISTANCE = {"near": 0.05, "mid": 0.35, "far": 0.75}
PAN = {"left": -0.6, "center": 0.0, "right": 0.6}
# planner models: all open weights with commercial-use licences (check the model card before shipping)
PLANNERS = {
    "qwen3vl-8b": dict(id="Qwen/Qwen3-VL-8B-Instruct", gb=18, lic="Apache-2.0"),
    "qwen3vl-4b": dict(id="Qwen/Qwen3-VL-4B-Instruct", gb=9, lic="Apache-2.0"),
    "qwen3vl-30b-a3b": dict(id="Qwen/Qwen3-VL-30B-A3B-Instruct", gb=62, lic="Apache-2.0"),
    "gemma4-26b": dict(id="google/gemma-4-26B-A4B-it", gb=52, lic="Apache-2.0"),
    "gemma4-31b": dict(id="google/gemma-4-31B-it", gb=62, lic="Apache-2.0"),
    "molmo2-8b": dict(id="allenai/Molmo2-8B", gb=18, lic="Apache-2.0", remote=True),
}
PLANNER_MODELS = {"8b": "qwen3vl-8b", "4b": "qwen3vl-4b", "2b": "qwen3vl-4b"}      # older names


# ------------------------------------------------------------------------------------------------ video io
def probe(path):
    """-> (duration_s, fps, width, height)"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return float(j["format"]["duration"]), float(num) / max(float(den), 1), int(s["width"]), int(s["height"])


def frames(path, fps=2.0, width=512, out_dir=None):
    """-> [(t_seconds, PIL.Image)] sampled at `fps` (ffmpeg's fps filter: frame k shows t = k / fps)."""
    from PIL import Image
    out = Path(out_dir or tempfile.mkdtemp(prefix="frames_"))
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("f_*.jpg"):
        f.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", f"fps={fps},scale={width}:-2",
                    "-q:v", "3", str(out / "f_%05d.jpg")], check=True)
    files = sorted(out.glob("f_*.jpg"))
    return [(k / fps, Image.open(f).convert("RGB")) for k, f in enumerate(files)]


def motion(path, fps=25.0, w=64, h=36):
    """Frame-difference energy per frame (tiny grey frames): peaks = visible hits, steps, flashes, cuts."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True, check=True)
    v = np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w).astype(np.float32)
    e = np.zeros(len(v), np.float32)
    if len(v) > 1:
        e[1:] = np.abs(np.diff(v, axis=0)).mean((1, 2))
    return e, fps


def snap(times, energy, fps, window=0.35):
    """Move each event time to the strongest motion peak within ±window s (keeps it if nothing stands out)."""
    out = []
    med = float(np.median(energy)) if len(energy) else 0.0
    for t in times:
        a, b = int(max(0, (t - window) * fps)), int(min(len(energy), (t + window) * fps + 1))
        if b - a < 2:
            out.append(t)
            continue
        w = energy[a:b]
        k = a + int(np.argmax(w >= 0.7 * w.max()))          # earliest strong change = the start of the action
        out.append(k / fps if energy[k] > 2.0 * med + 1e-6 else t)
    return out


# ------------------------------------------------------------------------------------------------ planner
DESCRIBE = """You are a game sound designer. The images are frames of a SILENT gameplay video, each preceded by its \
timestamp; the video is {dur:.1f} seconds long. Write a sound-design breakdown in plain text with four sections:
1. PLACE: where the scene happens, surface, weather, time of day, and what this place sounds like.
2. SOURCES: every visible thing that makes sound (characters, animals, vehicles, ships, weapons, objects, water, \
fire, machines): what it is, what it is made of, where it is in the frame (left / center / right), how far (near / \
mid / far).
3. TIMELINE: in time order, every action that makes a sound, with its time in seconds: each step, hit, shot, impact, \
jump, landing, door, spell, and when movements or engines start and stop. Count repeated actions.
4. OFF-SCREEN: sounds the place implies but the frames do not show (sea behind the camera, distant bells, wind, crowd).
Only what the frames support. No music, no speech."""

PLAN = """Below is a sound-design breakdown of a {dur:.1f}-second silent gameplay video. Turn it into a track plan.

BREAKDOWN:
{notes}

Return ONLY JSON:
{{"scene": "<one sentence>",
  "tracks": [
    {{"name": "<short name>",
      "kind": "ambience" | "continuous" | "events",
      "source": "<what makes the sound, e.g. 'a horse', 'heavy rain', 'an old wooden door'>",
      "action": "<what it does, e.g. 'galloping', 'falling', 'creaking open'>",
      "material": "<surface / material if it matters, e.g. 'on soft grass', 'on a tin roof', or ''>",
      "character": "<tone / quality, e.g. 'deep and heavy', 'distant and muffled', or ''>",
      "start": <s>, "end": <s>, "events": [<s>, ...], "event_seconds": <0.5-4>,
      "position": "left" | "center" | "right", "distance": "near" | "mid" | "far",
      "level": "low" | "medium" | "high"}}]}}
Rules:
- ambience: background bed of the place, usually 0 to {dur:.1f}; 1-3 of them.
- continuous: sounds without gaps while active (engine, galloping horse, river, fire, rain).
- events: separate short sounds; list the time of EACH occurrence (each shot of a gunfight, each sword hit, each \
bell strike). Footsteps are events when single steps are visible, otherwise continuous.
- 3-10 tracks. No music, no speech. Times between 0 and {dur:.1f}."""

DIST_WORDS = {"near": "close-up recording", "mid": "", "far": "heard from far away"}


def compose_prompt(t):
    """The text-to-audio prompt is built here from the planner's fields (source + action + material + character +
    distance): the planner describes, we write in the form Stable Audio 3 understands best."""
    src = str(t.get("source") or "").strip()
    if not src:
        return str(t.get("prompt") or t.get("name") or "").strip()
    act = str(t.get("action") or "").strip()
    head = f"{src[0].upper()}{src[1:]} {act}".strip()
    parts = [head]
    for k in ("material", "character"):
        v = str(t.get(k) or "").strip()
        if v:
            parts.append(v)
    d = DIST_WORDS.get(t.get("distance"), "")
    if d and d not in " ".join(parts):
        parts.append(d)
    if t.get("kind") == "ambience":
        parts.append("ambience")
    return ", ".join(parts)


class Planner:
    """A vision-language model watches the frames and writes the sound plan, in two steps (literature: splitting
    the reasoning and planning steps improves V2A planning, e.g. ThinkSound / PrismAudio):
      1. DESCRIBE  frames + timestamps -> free-text breakdown (place, sources, timeline, off-screen)
      2. PLAN      breakdown (text only, fast) -> JSON tracks with source / action / material / times
    The Stable Audio prompt is then composed by compose_prompt()."""

    def __init__(self, model="qwen3vl-8b", device=None, dtype=None, quant="auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        key = PLANNER_MODELS.get(model, model)
        spec = PLANNERS.get(key, dict(id=key, gb=20))
        mid, remote = spec["id"], bool(spec.get("remote"))
        self.name, self.model_id = key, mid
        cuda = torch.cuda.is_available()
        if dtype is None:
            dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
        kw = dict(dtype=dtype, device_map=device or ({"": 0} if cuda else "cpu"), trust_remote_code=remote)
        free = torch.cuda.mem_get_info()[0] / 2**30 if cuda else 0
        if quant == "4bit" or (quant == "auto" and cuda and spec["gb"] > 0.85 * free):
            self._bnb()
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype)
            print(f"planlayıcı 4-bit yükleniyor ({spec['gb']} GB > boş {free:.0f} GB)", flush=True)
        if remote:
            self._remote_deps()
        from . import hfcache
        # whole model on the GPU: "auto" silently puts layers on the CPU when memory is tight (minutes per plan)
        self.model = hfcache.load(lambda: AutoModelForImageTextToText.from_pretrained(mid, **kw).eval(), [mid])
        on = sorted({str(v) for v in (getattr(self.model, "hf_device_map", None) or {"": kw["device_map"]}).values()})
        if any("cpu" in d or "disk" in d for d in on) and cuda:
            print(f"UYARI: planlayıcının bir kısmı GPU dışında ({on}); planlama çok yavaş olur", flush=True)
        self.proc = hfcache.load(lambda: AutoProcessor.from_pretrained(mid, trust_remote_code=remote), [mid])
        hfcache.backup()
        print(f"planlayıcı: {mid} ({spec.get('lic', '?')})", flush=True)

    @staticmethod
    def _bnb():
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)

    @staticmethod
    def _remote_deps():
        import subprocess, sys
        for mod, pkg in [("molmo_utils", "molmo_utils"), ("decord", "decord")]:
            try:
                __import__(mod)
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    def ask(self, content, max_new_tokens=1536):
        import time
        import torch
        t0 = time.time()
        msgs = [{"role": "user", "content": content}]
        try:
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors="pt")
        except Exception:                                              # noqa: BLE001  (remote-code processors)
            images = [c["image"] for c in content if c.get("type") == "image"]
            text = self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"} if c.get("type") == "image" else c for c in content]}],
                tokenize=False, add_generation_prompt=True)
            inputs = self.proc(text=text, images=images or None, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        n_out = ids.shape[1] - n_in
        print(f"  planlayıcı: {n_in} girdi token'ı, {n_out} çıktı token'ı, {time.time() - t0:.0f} s"
              + (" (sınıra dayandı: çıktı yarım kalmış olabilir)" if n_out >= max_new_tokens else ""), flush=True)
        dec = getattr(self.proc, "batch_decode", None) or self.proc.tokenizer.batch_decode
        return dec(ids[:, n_in:], skip_special_tokens=True)[0]

    def describe(self, job, fps=2.0, width=448):
        content = []
        for t, img in job.frames(fps=fps, width=width):
            content += [{"type": "text", "text": f"<{t:.1f} seconds>"}, {"type": "image", "image": img}]
        content.append({"type": "text", "text": DESCRIBE.format(dur=job.duration)})
        print(f"planlayıcı 1/2: {len(content) // 2} kare ({fps:g} fps) izleniyor, sahne betimleniyor…", flush=True)
        return self.ask(content, max_new_tokens=1200).strip()

    def plan(self, job, fps=2.0, width=448, tries=2):
        job.notes = self.describe(job, fps=fps, width=width)
        print("planlayıcı 2/2: betimlemeden iz planı…", flush=True)
        last = None
        for k in range(tries):
            last = self.ask([{"type": "text", "text": PLAN.format(dur=job.duration, notes=job.notes)}])
            try:
                j = parse_json(last)
                job.scene = j.get("scene", "")
                tracks = j.get("tracks", [])
                for t in tracks:
                    t["prompt"] = compose_prompt(t)
                return clean(tracks, job.duration)
            except ValueError as e:
                print(f"  planlayıcı JSON vermedi ({e}); {'tekrar deneniyor' if k + 1 < tries else 'vazgeçildi'}",
                      flush=True)
        raise ValueError(f"planner did not return JSON:\n{last}")


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object")
    try:
        return json.loads(text[a: b + 1])
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e


def clean(tracks, dur):
    """Validate / clamp a plan (the planner's or a hand-edited one)."""
    out, names = [], set()
    for t in tracks:
        kind = t.get("kind", "ambience")
        kind = kind if kind in KINDS else "ambience"
        prompt = str(t.get("prompt") or t.get("name") or "").strip()
        if not prompt:
            continue
        name = str(t.get("name") or prompt[:24]).strip()
        while name in names:
            name += "+"
        names.add(name)
        start = float(np.clip(float(t.get("start", 0) or 0), 0, dur))
        end = float(np.clip(float(t.get("end", dur) or dur), start, dur))
        ev = sorted(float(np.clip(float(x), 0, dur)) for x in (t.get("events") or []) if _num(x))
        if kind == "events" and not ev:
            kind = "continuous"
        if kind != "events" and end - start < 0.5:
            start, end = max(0.0, start - 0.5), min(dur, end + 0.5)
        out.append(dict(name=name, kind=kind, prompt=prompt, start=start, end=end, events=ev if kind == "events" else [],
                        event_seconds=float(np.clip(float(t.get("event_seconds", 2) or 2), 0.5, 6)),
                        position=t.get("position") if t.get("position") in PAN else "center",
                        distance=t.get("distance") if t.get("distance") in DISTANCE else "mid",
                        level=t.get("level") if t.get("level") in LEVEL_DB else "medium",
                        seed=int(t.get("seed", len(out) + 1)), gain_db=float(t.get("gain_db", 0) or 0),
                        mute=bool(t.get("mute", False))))
        o = out[-1]
        o["pan"] = float(np.clip(float(t["pan"]) if _num(t.get("pan")) else PAN[o["position"]], -1, 1))
        o["dist"] = float(np.clip(float(t["dist"]) if _num(t.get("dist")) else DISTANCE[o["distance"]], 0, 1))
    return out


def _num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------ job
class Job:
    def __init__(self, video, work_dir, snap_events=True):
        self.video = Path(video)
        self.dir = Path(work_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.duration, self.fps, self.w, self.h = probe(self.video)
        self.scene, self.plan, self.snap_events = "", [], snap_events
        self.audio = {}                                  # track name -> stereo [2, T] at gen.sr (placed, mixed level)
        self.clips = {}                                  # track name -> generated audio (cached; reused by remix)
        self.candidates = 3                              # generated candidates per track
        self.scorer = None                               # e.g. ClapScorer(): picks the candidate closest to the prompt
        self.sr = None
        self._motion = None

    def frames(self, fps=2.0, width=448):
        return frames(self.video, fps, width, self.dir / "frames")

    def motion(self):
        if self._motion is None:
            self._motion = motion(self.video)
        return self._motion

    # -- plan io (editable by hand: plan.json)
    def save_plan(self):
        p = self.dir / "plan.json"
        p.write_text(json.dumps(dict(video=str(self.video), duration=self.duration, scene=self.scene,
                                     notes=getattr(self, "notes", ""), tracks=self.plan), indent=1, ensure_ascii=False))
        return p

    def load_plan(self, path=None):
        j = json.loads(Path(path or self.dir / "plan.json").read_text())
        self.scene, self.notes = j.get("scene", ""), j.get("notes", "")
        self.plan = clean(j["tracks"], self.duration)
        return self.plan

    def table(self):
        rows = []
        for i, t in enumerate(self.plan):
            when = (", ".join(f"{x:.1f}" for x in t["events"]) if t["kind"] == "events"
                    else f"{t['start']:.1f}–{t['end']:.1f} s")
            rows.append(f"{i + 1:2d}. {t['name']:<20s} {t['kind']:<10s} {when:<28s} {t['position']:<6s} "
                        f"{t['distance']:<4s} {t['level']:<6s} {t['gain_db']:+.0f} dB {'(sessiz) ' if t['mute'] else ''}| {t['prompt']}")
        return "\n".join(rows)

    # -- render: generation (slow, model) is cached per track; placement + mix (fast, DSP) is redone on every edit
    def track(self, key):
        """A track by name or by its 1-based number in table()."""
        if isinstance(key, int) or (isinstance(key, str) and key.strip().isdigit()):
            i = int(key) - 1
            if 0 <= i < len(self.plan):
                return self.plan[i]
        t = next((x for x in self.plan if x["name"] == str(key).strip()), None)
        if t is None:
            raise KeyError(f"iz bulunamadı: '{key}'. İzler: "
                           + ", ".join(f"{i + 1} = {x['name']}" for i, x in enumerate(self.plan)))
        return t

    @staticmethod
    def _sig(t, sr):
        """What the generated audio depends on (anything else is mixing)."""
        if t["kind"] == "events":
            return (t["kind"], t["prompt"], t["seed"], round(t["event_seconds"], 2), len(t["events"]))
        return (t["kind"], t["prompt"], t["seed"], int(round((t["end"] - t["start"]) * 2)))

    def _generate(self, gen, t):
        """k candidates per track (k = self.candidates); with a scorer (CLAP) the best one goes into the mix, the
        others stay selectable (pick). Events: k x n clips, the n best form candidate 1, the next n candidate 2 ..."""
        k = max(1, int(self.candidates))
        sig = self._sig(t, gen.sr) + (k,)
        c = self.clips.get(t["name"])
        if c is not None and c["sig"] == sig:
            return False
        if t["kind"] == "events":
            # Never ask the model for very short clips: at 1-2 s Stable Audio 3 largely ignores the prompt (04's
            # reference set: 1 s sounds matched their prompt worst). Generate >= 4 s, trim to the sound afterwards.
            sec, n = max(EVENT_GEN_S, t["event_seconds"] + 1.0), len(t["events"])
        else:
            sec, n = min(t["end"] - t["start"] + 0.5, 120.0), 1
        clips = _batched(gen, t["prompt"], sec, n * k, t["seed"])
        scores = self.scorer(t["prompt"], clips, gen.sr) if self.scorer else None
        if scores is not None:
            order = np.argsort(-np.asarray(scores))
            clips, scores = [clips[j] for j in order], [float(scores[j]) for j in order]
        sets = [clips[j * n:(j + 1) * n] for j in range(k)]
        set_scores = [float(np.mean(scores[j * n:(j + 1) * n])) for j in range(k)] if scores is not None else None
        self.clips[t["name"]] = dict(sig=sig, sets=sets, scores=set_scores, pick=0, clips=sets[0])
        sc = (" · CLAP " + " / ".join(f"{x:.2f}" for x in set_scores)) if set_scores else ""
        print(f"    istek: “{t['prompt']}” · tohum {t['seed']} · {k} aday{sc}", flush=True)
        return True

    def pick(self, key, j):
        """Use candidate j (0-based) of a track; mixing only."""
        t = self.track(key)
        c = self.clips.get(t["name"])
        if not c or not 0 <= j < len(c["sets"]):
            raise ValueError("bu iz için böyle bir aday yok")
        c["pick"], c["clips"] = j, c["sets"][j]
        return self.remix()

    def render(self, gen, only=None, force=False):
        """Generate what is missing / changed (all tracks, or only those in `only`), then place + mix + mux."""
        import time
        self.sr = gen.sr
        t0 = time.time()
        made = 0
        for t in self.plan:
            if only and t["name"] not in only:
                continue
            if force:
                self.clips.pop(t["name"], None)
            t1 = time.time()
            if not t["mute"] and self._generate(gen, t):
                made += 1
                print(f"  {t['name']:<20s} {t['kind']:<10s} üretildi ({time.time() - t1:.1f} s)", flush=True)
        t_gen = time.time() - t0
        out = self.remix()
        self.timing = dict(gen_s=t_gen, tracks=made, **self.timing)
        print(f"  süre: ses üretimi {t_gen:.1f} s ({made} iz) · " + self.timing_text(), flush=True)
        return out

    def timing_text(self):
        t = getattr(self, "timing", {})
        return " · ".join(f"{k} {t[v]:.1f} s" for k, v in (("yerleştirme", "place_s"), ("miks", "mix_s"),
                                                            ("video", "mux_s"), ("çizelge", "timeline_s")) if v in t)

    def remix(self):
        """Mixing only (level, mute, pan, distance, times): no model call, ~1 s. -> mp4 path."""
        if self.sr is None:
            raise RuntimeError("önce render(gen)")
        n = int(round(self.duration * self.sr))
        if self.snap_events:
            e, f = self.motion()
        import time
        t0 = time.time()
        self.audio, self.placed = {}, {}                 # audio: what goes into the mix; placed: the track itself
        for t in self.plan:
            if t["kind"] == "events" and self.snap_events:
                t["events_snapped"] = snap(t["events"], e, f)
            c = self.clips.get(t["name"])
            pl = self._place(t, c["clips"], n) if c and c["clips"] else np.zeros((2, n), np.float32)
            self.placed[t["name"]] = pl
            self.audio[t["name"]] = np.zeros_like(pl) if t["mute"] else pl
        t1 = time.time()
        self.mixdown()
        t2 = time.time()
        out = self.mux()
        self.timing = dict(place_s=t1 - t0, mix_s=t2 - t1, mux_s=time.time() - t2)
        return out

    def _place(self, t, clips, n):
        from .fx import Rack, process
        sr = self.sr
        buf = np.zeros((2, n), np.float32)
        rack = Rack(distance=t["dist"], pan=t["pan"])
        if t["kind"] in ("ambience", "continuous"):
            a, b = int(t["start"] * sr), int(t["end"] * sr)
            seg = b - a
            clip = _loop(clips[0], seg, sr)
            if t["kind"] == "continuous":                          # point source: mono -> distance + pan
                st, _ = process(clip.mean(0), sr, rack)
                clip = st[:, :seg]
            else:                                                  # bed: its own stereo; distance darkens, pan tilts
                if t["dist"] > 0.5:
                    clip = _lowpass(clip, sr, 9000 - 7000 * t["dist"])
                if abs(t["pan"]) > 0.01:
                    g = np.array([np.sqrt(0.5 * (1 - t["pan"])), np.sqrt(0.5 * (1 + t["pan"]))]) * np.sqrt(2)
                    clip = clip * g[:, None].astype(np.float32)
            fi = min(int(0.4 * sr), seg // 4) if a > 0 else int(0.05 * sr)
            fo = min(int(0.6 * sr), seg // 4) if b < n else int(0.05 * sr)
            clip = _fade_io(clip[:, :seg], fi, fo)
            buf[:, a: a + clip.shape[1]] += clip
        else:
            times = t.get("events_snapped") or t["events"]
            for tm, c in zip(times, clips):
                st, _ = process(c.mean(0), sr, rack)
                st = st[:, _onset(st, sr):]                       # the sound's attack lands on the event time
                st = _trim_tail(st, sr, t["event_seconds"])       # its own decay, not the rest of the long clip
                a = int(tm * sr)
                k = min(st.shape[1], n - a)
                if k > 0:
                    buf[:, a: a + k] += _fade_io(st[:, :k], 0, min(int(0.02 * sr), k // 4))
        return _norm_rms(buf, LEVEL_DB[t["level"]] + t["gain_db"])

    def mixdown(self):
        from .fx import limiter
        import soundfile as sf
        stems = self.dir / "stems"
        stems.mkdir(exist_ok=True)
        for f in stems.glob("*.wav"):
            f.unlink()
        mix = None
        self.stems = []
        for i, (name, a) in enumerate(self.audio.items()):
            # each stem is the track itself, also when muted; a new name per mix so players never replay a cached file
            f = stems / f"{i + 1:02d}_{_slug(name)}_{getattr(self, '_ver', 0) + 1:03d}.wav"
            sf.write(f, getattr(self, "placed", {}).get(name, a).T, self.sr, subtype="PCM_24")
            self.stems.append(f)
            mix = a.copy() if mix is None else mix + a
        if mix is None:
            raise ValueError("plan boş")
        mix = limiter(mix * (10 ** (-1 / 20)))
        self.mix_wav = self.dir / "mix.wav"
        sf.write(self.mix_wav, np.asarray(mix).T, self.sr, subtype="PCM_24")
        return self.mix_wav

    def timeline(self, solo=None):
        """PNG: one lane per track (loudness over time), event marks, muted lanes grey. -> path"""
        import time
        t_start = time.time()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(self.audio)
        h = 0.55 * max(len(names), 1) + 0.6
        fig, ax = plt.subplots(figsize=(9, h), dpi=110)
        hop = int(0.05 * self.sr)
        cols = ["#e8873a", "#3a8fe8", "#4fb36b", "#c25bd6", "#d6c25b", "#5bd6c8", "#d65b6e", "#8a8f99"]
        for k, name in enumerate(names):
            t = next(x for x in self.plan if x["name"] == name)
            a = getattr(self, "placed", self.audio).get(name, self.audio[name]).mean(0)
            m = len(a) // hop
            env = np.sqrt((a[: m * hop].reshape(m, hop) ** 2).mean(1) + 1e-12)
            db = np.clip((20 * np.log10(env) + 60) / 60, 0, 1) * 0.42
            x = np.arange(m) * hop / self.sr
            y0 = len(names) - 1 - k
            off = t["mute"] or (solo and solo != name)
            c = "#9a9ca3" if off else cols[k % len(cols)]
            ax.fill_between(x, y0 - db, y0 + db, color=c, alpha=0.35 if off else 0.85, lw=0)
            for e in (t.get("events_snapped") or t["events"]) if t["kind"] == "events" else []:
                ax.plot([e, e], [y0 - 0.45, y0 + 0.45], color=c, lw=1.2)
            ax.text(-0.01 * self.duration, y0, f"{k + 1}. {name}", ha="right", va="center", fontsize=8,
                    color="#9a9ca3" if off else "#333")
        ax.set_xlim(0, self.duration)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_yticks([])
        ax.set_xlabel("saniye", fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.6 / h)
        out = self.dir / "timeline.png"
        fig.savefig(out)
        getattr(self, "timing", {}).update(timeline_s=time.time() - t_start)
        plt.close(fig)
        return out

    def mux(self, out=None):
        """Video stream copied as is + the mix as AAC (any existing audio in the input is dropped).
        A new file name each time (players / browsers cache by name); older versions are removed."""
        self._ver = getattr(self, "_ver", 0) + 1
        old = list(self.dir.glob(f"{self.video.stem}_sesli*.mp4"))
        self.out_mp4 = Path(out or self.dir / f"{self.video.stem}_sesli_{self._ver:03d}.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.video), "-i", str(self.mix_wav), "-map", "0:v:0",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        str(self.out_mp4)], check=True)
        for f in old:
            if f != self.out_mp4:
                f.unlink(missing_ok=True)
        return self.out_mp4

    def edit(self, key, **changes):
        """Change a track (gain_db, mute, pan, dist, level, prompt, seed, events, start, end, ...).
        Returns the track. Call remix() (mixing changes) or render(gen) (prompt / seed / length / count changes)."""
        t = self.track(key)
        name = t["name"]
        t.update({k: v for k, v in changes.items() if v is not None})
        i = self.plan.index(t)
        self.plan = clean(self.plan, self.duration)
        if self.plan[i]["name"] != name:                          # keep the cache when the name was normalised
            self.clips[self.plan[i]["name"]] = self.clips.pop(name, None) or {"sig": None, "clips": []}
        self.save_plan()
        return self.plan[i]

    def regenerate(self, gen, key, prompt=None, seed=None, **changes):
        """Re-do one track (new prompt and / or new seed); the other tracks are not touched. -> mp4"""
        t = self.track(key)
        t = self.edit(t["name"], prompt=prompt or None, seed=int(seed) if seed is not None else t["seed"] + 100,
                      **changes)
        return self.render(gen, only=[t["name"]])

    def add(self, gen, **track):
        """Add a track by hand (e.g. the planner missed the church bell) and render only that one."""
        self.plan = clean(self.plan + [track], self.duration)
        self.save_plan()
        return self.render(gen, only=[self.plan[-1]["name"]])

    def remove(self, key):
        t = self.track(key)
        self.plan.remove(t)
        self.clips.pop(t["name"], None)
        self.save_plan()
        return self.remix()

    def package(self, name=None):
        """zip: video with sound, mix, stems, plan."""
        import zipfile
        self.save_plan()
        z = self.dir / f"{name or self.video.stem}_paket.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in [self.out_mp4, self.mix_wav, self.dir / "plan.json", *self.stems]:
                zf.write(f, f.relative_to(self.dir))
        return z


# ------------------------------------------------------------------------------------------------ dsp helpers
def _loop(clip, n, sr, xf=0.5):
    """Make a stereo clip exactly n samples long: crossfaded repeats if it is too short."""
    if clip.shape[1] >= n:
        return clip[:, :n]
    k = int(min(xf * sr, clip.shape[1] // 3))
    out = clip.copy()
    ramp = np.linspace(0, 1, k, dtype=np.float32)
    while out.shape[1] < n:
        head = out[:, -k:] * (1 - ramp) + clip[:, :k] * ramp
        out = np.concatenate([out[:, :-k], head, clip[:, k:]], axis=1)
    return out[:, :n]


def _fade_io(x, fi, fo):
    x = x.copy()
    if fi > 0:
        x[:, :fi] *= np.linspace(0, 1, fi, dtype=np.float32)
    if fo > 0:
        x[:, -fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return x


def _onset(x, sr, rel_db=-24.0):
    """First sample within the first 1 s where the envelope reaches peak + rel_db (minus 5 ms pre-roll)."""
    m = np.abs(x).max(0)
    lim = m[: int(sr)]
    if lim.size == 0 or lim.max() <= 0:
        return 0
    k = int(np.argmax(lim >= m.max() * 10 ** (rel_db / 20)))
    return max(0, k - int(0.005 * sr))


def _batched(gen, prompt, seconds, n, seed, max_batch=16):
    out = []
    for k in range(0, n, max_batch):
        out += list(gen.raw(prompt, seconds=seconds, n=min(max_batch, n - k), seed=seed + 1000 * (k // max_batch)))
    return out


class ClapScorer:
    """Text-audio similarity with LAION CLAP (laion/clap-htsat-unfused; evaluation model, not the generator's text
    encoder). Deterministic: scores the first 10 s of each clip (mono)."""

    def __init__(self, device=None):
        from .evaluate import Clap
        self.clap = Clap(device)

    def __call__(self, text, clips, sr):
        wavs = [np.asarray(c, np.float32).mean(0)[: int(10 * sr)] for c in clips]
        a = np.concatenate([self.clap.audio(wavs[k:k + 16], sr) for k in range(0, len(wavs), 16)])
        return a @ self.clap.text([text])[0]


def _trim_tail(x, sr, seconds, floor_db=-40.0, fade_s=0.25):
    """Keep at least `seconds`, then until the sound has decayed to peak + floor_db (max the clip), fade out."""
    n = x.shape[1]
    keep = min(n, int(seconds * sr))
    env = np.abs(x).max(0)
    pk = env.max()
    if pk > 0 and keep < n:
        hop = int(0.02 * sr)
        m = (n - keep) // hop
        if m > 0:
            frames = env[keep: keep + m * hop].reshape(m, hop).max(1)
            quiet = np.nonzero(frames < pk * 10 ** (floor_db / 20))[0]
            keep += int(quiet[0] * hop) if len(quiet) else m * hop
    k = min(int(fade_s * sr), keep // 3)
    return _fade_io(x[:, :keep], 0, k)


def _lowpass(x, sr, fc):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, fc, fs=sr, output="sos"), x, axis=-1).astype(np.float32)


def _norm_rms(x, target_db):
    """Scale so the track's active part (frames above -50 dBFS) has the target RMS."""
    m = np.abs(x).max(0)
    act = x[:, m > 10 ** (-50 / 20)]
    if act.size == 0:
        return x
    rms = float(np.sqrt((act ** 2).mean()))
    return (x * (10 ** (target_db / 20) / max(rms, 1e-9))).astype(np.float32)


def _slug(s):
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s.translate(tr).strip().lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep[:40].strip("_") or "iz"


def demo_video(path, seconds=12.0):
    """A synthetic test clip (moving box that 'hits' at 2, 5, 8 s) for checking the pipeline without real footage."""
    hits = "+".join(f"between(t,{h},{h + 0.12})" for h in (2, 5, 8))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x203040:s=640x360:d={seconds}:r=25",
                    "-f", "lavfi", "-i", f"color=c=orange:s=60x60:d={seconds}:r=25", "-filter_complex",
                    f"[0][1]overlay=x='40+mod(t*60,520)':y=150,drawbox=x=0:y=0:w=640:h=360:color=white:t=fill:"
                    f"enable='{hits}'", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return Path(path)


def ffmpeg_ok():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
