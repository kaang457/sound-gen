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
    job.render(gen)                                               # -> job.mix_wav, job.stems
    job.mux()                                                     # -> job.out_mp4
    job.regenerate(gen, "Sea waves", prompt="Calm sea waves gently lapping on a rocky shore", seed=7)
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

KINDS = ("ambience", "continuous", "events")
LEVEL_DB = {"low": -18.0, "medium": -10.0, "high": -4.0}
DISTANCE = {"near": 0.05, "mid": 0.35, "far": 0.75}
PAN = {"left": -0.6, "center": 0.0, "right": 0.6}
PLANNER_MODELS = {"8b": "Qwen/Qwen3-VL-8B-Instruct", "4b": "Qwen/Qwen3-VL-4B-Instruct",
                  "2b": "Qwen/Qwen3-VL-2B-Instruct"}


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
PROMPT = """You are a game sound designer. The images are frames of a SILENT gameplay video, each preceded by its \
timestamp. The video is {dur:.1f} seconds long. List every sound this scene needs: on-screen sources (characters, \
footsteps, horses, weapons, vehicles, ships, doors, impacts, magic) and off-screen ambience implied by the place \
(sea, wind, rain, city, forest, church bells far away, crowd).

Return ONLY JSON, no prose:
{{"scene": "<one sentence>",
  "tracks": [
    {{"name": "<short name>",
      "kind": "ambience" | "continuous" | "events",
      "prompt": "<English sound-effect description for a text-to-audio model: source, action, material, recording character; no music, no speech>",
      "start": <seconds>, "end": <seconds>,
      "events": [<seconds>, ...],
      "event_seconds": <length of one event sound, 0.5-4>,
      "position": "left" | "center" | "right",
      "distance": "near" | "mid" | "far",
      "level": "low" | "medium" | "high"}}
  ]}}

Rules:
- ambience: background bed for the whole place (usually start 0, end {dur:.1f}); 1-3 ambience tracks.
- continuous: a source that sounds without gaps while visible or active (engine, horse galloping, river, fire).
- events: separate short sounds at exact moments (gunshot, sword hit, door slam, bell strike, horn blast, jump \
landing). Give the time of EACH occurrence in "events", as precisely as the frames allow. Footsteps are events \
(one per step) when steps are visible; otherwise continuous.
- Several shots in a gunfight are several events of one track, not one long sound.
- Only sounds a player would expect; no music, no dialogue. 3-10 tracks in total.
- Times are seconds from the start of the video, between 0 and {dur:.1f}."""


class Planner:
    """Qwen3-VL (Apache-2.0) watches the frames and writes the sound plan."""

    def __init__(self, model="8b", device=None, dtype=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        mid = PLANNER_MODELS.get(model, model)
        self.model_id = mid
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(mid, dtype=dtype, device_map=device or "auto").eval()
        self.proc = AutoProcessor.from_pretrained(mid)

    def ask(self, content, max_new_tokens=2048):
        import torch
        msgs = [{"role": "user", "content": content}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def plan(self, job, fps=2.0, width=448, tries=2):
        content = []
        for t, img in job.frames(fps=fps, width=width):
            content += [{"type": "text", "text": f"<{t:.1f} seconds>"}, {"type": "image", "image": img}]
        content.append({"type": "text", "text": PROMPT.format(dur=job.duration)})
        last = None
        for _ in range(tries):
            last = self.ask(content)
            try:
                j = parse_json(last)
                job.scene = j.get("scene", "")
                return clean(j.get("tracks", []), job.duration)
            except ValueError:
                continue
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
                                     tracks=self.plan), indent=1, ensure_ascii=False))
        return p

    def load_plan(self, path=None):
        j = json.loads(Path(path or self.dir / "plan.json").read_text())
        self.scene = j.get("scene", "")
        self.plan = clean(j["tracks"], self.duration)
        return self.plan

    def table(self):
        rows = []
        for i, t in enumerate(self.plan):
            when = (", ".join(f"{x:.1f}" for x in t["events"]) if t["kind"] == "events"
                    else f"{t['start']:.1f}–{t['end']:.1f} s")
            rows.append(f"{i + 1:2d}. {t['name']:<20s} {t['kind']:<10s} {when:<28s} {t['position']:<6s} "
                        f"{t['distance']:<4s} {t['level']:<6s} {'(sessiz) ' if t['mute'] else ''}| {t['prompt']}")
        return "\n".join(rows)

    # -- render
    def render(self, gen, only=None, variations=None):
        """Generate the tracks (all, or only the names in `only`) and write stems + mix."""
        self.sr = gen.sr
        n = int(round(self.duration * self.sr))
        if self.snap_events:
            e, f = self.motion()
        for t in self.plan:
            if only and t["name"] not in only:
                continue
            if t["mute"]:
                self.audio[t["name"]] = np.zeros((2, n), np.float32)
                continue
            if t["kind"] == "events" and self.snap_events:
                t["events_snapped"] = snap(t["events"], e, f)
            self.audio[t["name"]] = self._track(gen, t, n)
            print(f"  {t['name']:<20s} {t['kind']:<10s} hazır")
        self.audio = {t["name"]: self.audio[t["name"]] for t in self.plan if t["name"] in self.audio}
        return self.mixdown()

    def _track(self, gen, t, n):
        from .fx import Rack, process
        sr = gen.sr
        buf = np.zeros((2, n), np.float32)
        if t["kind"] in ("ambience", "continuous"):
            a, b = int(t["start"] * sr), int(t["end"] * sr)
            seg = b - a
            clip = gen.raw(t["prompt"], seconds=min(seg / sr + 0.5, 120.0), n=1, seed=t["seed"])[0]
            clip = _loop(clip, seg, sr)
            if t["kind"] == "continuous":                          # point source: mono -> distance + pan
                st, _ = process(clip.mean(0), sr, Rack(distance=DISTANCE[t["distance"]], pan=PAN[t["position"]]))
                clip = st[:, :seg]
            else:                                                  # bed: keep its own stereo, far = darker
                if t["distance"] == "far":
                    clip = _lowpass(clip, sr, 3500)
            fi = min(int(0.4 * sr), seg // 4) if a > 0 else int(0.05 * sr)
            fo = min(int(0.6 * sr), seg // 4) if b < n else int(0.05 * sr)
            clip = _fade_io(clip[:, :seg], fi, fo)
            buf[:, a: a + clip.shape[1]] += clip
        else:
            times = t.get("events_snapped") or t["events"]
            clips = gen.raw(t["prompt"], seconds=t["event_seconds"], n=len(times), seed=t["seed"])  # all different
            for tm, c in zip(times, clips):
                st, _ = process(c.mean(0), sr, Rack(distance=DISTANCE[t["distance"]], pan=PAN[t["position"]]))
                st = st[:, _onset(st, sr):]                       # the sound's attack lands on the event time
                a = int(tm * sr)
                k = min(st.shape[1], n - a)
                if k > 0:
                    buf[:, a: a + k] += _fade_io(st[:, :k], 0, min(int(0.02 * sr), k // 4))
        buf = _norm_rms(buf, LEVEL_DB[t["level"]] + t["gain_db"])
        return buf

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
            f = stems / f"{i + 1:02d}_{_slug(name)}.wav"
            sf.write(f, a.T, self.sr, subtype="PCM_24")
            self.stems.append(f)
            mix = a.copy() if mix is None else mix + a
        if mix is None:
            raise ValueError("plan is empty")
        mix = limiter(mix * (10 ** (-1 / 20)))
        self.mix_wav = self.dir / "mix.wav"
        sf.write(self.mix_wav, np.asarray(mix).T, self.sr, subtype="PCM_24")
        return self.mix_wav

    def mux(self, out=None):
        """Video stream copied as is + the mix as AAC (any existing audio in the input is dropped)."""
        self.out_mp4 = Path(out or self.dir / f"{self.video.stem}_sesli.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.video), "-i", str(self.mix_wav), "-map", "0:v:0",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
                        str(self.out_mp4)], check=True)
        return self.out_mp4

    def regenerate(self, gen, name, prompt=None, seed=None, **changes):
        """Re-do one track (new prompt / seed / level / times ...) and re-render the mix; others stay as they are."""
        t = next((x for x in self.plan if x["name"] == name), None)
        if t is None:
            raise KeyError(f"no track '{name}'; tracks: {[x['name'] for x in self.plan]}")
        if prompt:
            t["prompt"] = prompt
        t["seed"] = int(seed) if seed is not None else t["seed"] + 100
        t.update(changes)
        self.plan = clean(self.plan, self.duration)
        self.render(gen, only=[name])
        return self.mux()

    def add(self, gen, **track):
        """Add a track by hand (e.g. the planner missed the church bell) and render only that one."""
        self.plan = clean(self.plan + [track], self.duration)
        self.render(gen, only=[self.plan[-1]["name"]])
        return self.mux()

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
