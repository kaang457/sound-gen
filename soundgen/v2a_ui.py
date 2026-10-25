"""Video sound studio (Gradio, runs in Colab with a share link).

    upload a silent gameplay clip → "Sahneyi çöz" (Qwen3-VL plan + Stable Audio 3 tracks) → video with sound
    every track has its own row:  volume (dB) · mute · solo · pan · distance · times · prompt → "Yeniden üret"
    mixing changes (volume, mute, solo, pan, distance, event times) never call the model: ~1 s
    prompt / new variation calls the model for that one track only
    add a track the planner missed, remove a track, download the package (mp4 + mix + stems + plan)

    from soundgen.v2a_ui import build
    build(gen, planner, P.drive / "v2a").queue().launch(share=True)

VideoStudio holds the logic (no Gradio inside, testable); build() only wires the widgets.
"""
import re
import shutil
import time
from pathlib import Path

from . import v2a

N_SLOTS = 12
KIND_TR = {"ambience": "ortam", "continuous": "sürekli", "events": "olay"}
KIND_FROM = {v: k for k, v in KIND_TR.items()}


def times_text(t):
    if t["kind"] == "events":
        return ", ".join(f"{x:.2f}".rstrip("0").rstrip(".") for x in t["events"])
    return f"{t['start']:.1f}-{t['end']:.1f}"


RANGE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)\s*(s|sn)?\s*$")


def parse_times(text, kind, dur):
    """The form of the text decides, whatever the track was before:
         '0-12', '3.5 – 9'      -> a span: start / end (an event track becomes 'continuous')
         '2, 5.1, 8', '6.5'     -> event times (a span track becomes 'events')
       '2,5' (no space) is 2.5."""
    text = (text or "").strip()
    m = RANGE.match(text)
    if m:
        a, b = sorted(min(max(float(x.replace(",", ".")), 0.0), dur) for x in m.groups()[:2])
        if b - a < 0.2:
            raise ValueError("aralık çok kısa")
        return dict(start=a, end=b, kind=kind if kind in ("ambience", "continuous") else "continuous")
    nums = [min(max(float(x.replace(",", ".")), 0.0), dur) for x in re.findall(r"\d+(?:[.,]\d+)?", text)]
    if not nums:
        raise ValueError("zaman yaz: olay anları '2, 5.1, 8' ya da aralık '0-12'")
    return dict(events=sorted(nums), kind="events")


class VideoStudio:
    def __init__(self, gen, planner, root):
        self.gen, self.planner = gen, planner
        self.root = Path(root)
        self.job = None
        self.solo = None

    # ---------------------------------------------------------------- whole video
    def open(self, video_path, plan_fps=2.0):
        src = Path(video_path)
        name = v2a._slug(src.stem) or "video"
        work = self.root / f"{name}_{time.strftime('%m%d_%H%M%S')}"
        work.mkdir(parents=True, exist_ok=True)
        dst = work / f"{name}{src.suffix.lower() or '.mp4'}"
        shutil.copy2(src, dst)
        self.job = v2a.Job(dst, work)
        self.solo = None
        t0 = time.time()
        self.job.plan = self.planner.plan(self.job, fps=plan_fps)
        self.job.save_plan()
        t1 = time.time()
        print(f"plan hazır ({t1 - t0:.0f} s): {len(self.job.plan)} iz; sesler üretiliyor…", flush=True)
        out = self.job.render(self.gen)
        return out, (f"**Sahne:** {self.job.scene or '–'}  \n{len(self.job.plan)} iz · plan {t1 - t0:.0f} s · "
                     f"ses {time.time() - t1:.0f} s · video {self.job.duration:.1f} s")

    def regenerate_all(self):
        self._need()
        for t in self.job.plan:
            t["seed"] += 100
        return self.job.render(self.gen), "bütün izler yeniden üretildi"

    def package(self):
        self._need()
        return str(self.job.package())

    # ---------------------------------------------------------------- one track
    def slot(self, i):
        """Values for slot i (dict) or None if the slot is empty."""
        if not self.job or i >= len(self.job.plan):
            return None
        t = self.job.plan[i]
        stem = self.job.stems[i] if getattr(self.job, "stems", None) and i < len(self.job.stems) else None
        when = (f"{len(t['events'])} olay" if t["kind"] == "events" else f"{t['start']:.1f}–{t['end']:.1f} s")
        flag = " · 🔇" if t["mute"] else (" · 🎧 solo" if self.solo == t["name"] else "")
        return dict(header=f"**{i + 1}. {t['name']}** · {KIND_TR[t['kind']]} · {when}{flag}", prompt=t["prompt"],
                    times=times_text(t), gain=t["gain_db"], mute=t["mute"], pan=t["pan"], dist=t["dist"],
                    solo=self.solo == t["name"], stem=str(stem) if stem else None)

    def set_mix(self, i, gain=None, mute=None, pan=None, dist=None):
        self._need()
        t = self.job.edit(i + 1, gain_db=gain, mute=mute, pan=pan, dist=dist)
        if not t["mute"] and t["name"] not in self.job.clips:      # unmuted a track that was never generated
            return self.job.render(self.gen, only=[t["name"]]), f"{t['name']}: üretildi"
        return self._remix(), f"{t['name']}: miks güncellendi"

    def set_times(self, i, text):
        self._need()
        t = self.job.plan[i]
        if (text or "").strip() == times_text(t):
            return self.job.out_mp4, f"{t['name']}: zaman aynı"
        ch = parse_times(text, t["kind"], self.job.duration)
        was = t["kind"]
        t = self.job.edit(i + 1, **ch)
        note = f" (tür: {KIND_TR[was]} → {KIND_TR[t['kind']]})" if t["kind"] != was else ""
        return self.job.render(self.gen, only=[t["name"]]), f"{t['name']}: zaman {times_text(t)}{note}"

    def regenerate(self, i, prompt=None, times=None):
        """New variation of one track with what is in its row now (prompt and time); others untouched."""
        self._need()
        t = self.job.plan[i]
        if times is not None and times.strip() and times.strip() != times_text(t):
            self.job.edit(i + 1, **parse_times(times, t["kind"], self.job.duration))
            t = self.job.plan[i]
        p = (prompt or "").strip()
        out = self.job.regenerate(self.gen, i + 1, prompt=p if p and p != t["prompt"] else None)
        t = self.job.plan[i]
        return out, (f"{t['name']}: yeni varyasyon · {KIND_TR[t['kind']]} {times_text(t)} · "
                     f"“{t['prompt']}” · tohum {t['seed']}")

    def toggle_solo(self, i):
        self._need()
        name = self.job.plan[i]["name"]
        self.solo = None if self.solo == name else name
        return self._remix(), (f"solo: {name}" if self.solo else "solo kapalı")

    def remove(self, i):
        self._need()
        name = self.job.plan[i]["name"]
        if self.solo == name:
            self.solo = None
        self.job.remove(i + 1)
        return self._remix(), f"{name} silindi"

    def add(self, name, kind_tr, prompt, times, position="center"):
        self._need()
        if not (prompt or "").strip():
            raise ValueError("yeni iz için istek yaz")
        if len(self.job.plan) >= N_SLOTS:
            raise ValueError(f"en fazla {N_SLOTS} iz")
        kind = KIND_FROM.get(kind_tr, kind_tr)
        tr = dict(name=(name or prompt[:24]).strip(), kind=kind, prompt=prompt.strip(), position=position,
                  seed=int(time.time()) % 100000)
        tr.update(parse_times(times or (f"0-{self.job.duration:.1f}" if kind != "events" else ""), kind,
                              self.job.duration))
        out = self.job.add(self.gen, **tr)
        return out, f"{self.job.plan[-1]['name']} eklendi"

    # ---------------------------------------------------------------- helpers
    def _remix(self):
        if self.solo:
            keep = {t["name"]: t["mute"] for t in self.job.plan}
            for t in self.job.plan:
                t["mute"] = t["name"] != self.solo
            try:
                return self.job.remix()
            finally:
                for t in self.job.plan:
                    t["mute"] = keep[t["name"]]
                self.job.save_plan()
        return self.job.remix()

    def _need(self):
        if self.job is None:
            raise ValueError("önce bir video yükle ve 'Sahneyi çöz'e bas")


CSS = """
.gradio-container {max-width: 1280px !important}
.trk {border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important; padding: 6px 8px !important;
      margin-bottom: 6px !important}
.trk .prose p {margin: 0 !important}
#go, #add {font-weight: 700}
"""


def build(gen, planner, root, title="Video Ses Stüdyosu"):
    import gradio as gr
    st = VideoStudio(gen, planner, root)

    def err(e):
        raise gr.Error(str(e))

    with gr.Blocks(title=title, theme=gr.themes.Base(primary_hue="orange", neutral_hue="zinc"), css=CSS) as demo:
        gr.Markdown(f"# {title}\nSessiz oyun videosu yükle → **Sahneyi çöz** → her izi ayrı düzenle: ses seviyesi, "
                    "sustur, solo, stereo konum, uzaklık, zamanlar, istek → **Yeniden üret**. Miks değişiklikleri "
                    "anında (model çağırmaz); yeniden üretme yalnız o izi değiştirir.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                vin = gr.Video(label="Sessiz video (10–30 s)", sources=["upload"])
                with gr.Row():
                    go = gr.Button("Sahneyi çöz ve sesle", variant="primary", elem_id="go")
                    again = gr.Button("Hepsini yeniden üret")
                vout = gr.Video(label="Sesli video", interactive=False, autoplay=False)
                status = gr.Markdown("")
                tl = gr.Image(label="Zaman çizelgesi: hangi iz ne zaman ses çıkarıyor (çizgiler: olay anları)",
                              interactive=False, type="filepath")
                with gr.Row():
                    pack = gr.Button("Paketi indir (mp4 + miks + izler + plan)")
                    pfile = gr.File(label="Paket")
                with gr.Accordion("İz ekle (planlayıcının kaçırdığı ses)", open=False):
                    a_name = gr.Textbox(label="Ad", placeholder="ör. Kilise çanı")
                    a_kind = gr.Radio(list(KIND_FROM), value="olay", label="Tür")
                    a_prompt = gr.Textbox(label="İstek (İngilizce)", placeholder="A large church bell rings once, far away")
                    a_times = gr.Textbox(label="Zaman", placeholder="olay: 3, 9 · ortam/sürekli: 0-12")
                    a_pos = gr.Radio(["left", "center", "right"], value="center", label="Konum")
                    add_btn = gr.Button("Ekle", elem_id="add")
            with gr.Column(scale=6):
                gr.Markdown("### İzler")
                rows = []
                for i in range(N_SLOTS):
                    with gr.Group(visible=False, elem_classes="trk") as box:
                        head = gr.Markdown("")
                        with gr.Row():
                            gain = gr.Slider(-30, 12, 0, step=1, label="Ses (dB)", scale=3)
                            pan = gr.Slider(-1, 1, 0, step=0.05, label="Sol ↔ sağ", scale=3)
                            dist = gr.Slider(0, 1, 0.3, step=0.05, label="Yakın → uzak", scale=3)
                        with gr.Row():
                            mute = gr.Checkbox(label="Sustur", scale=1)
                            solo = gr.Button("Solo", size="sm", scale=1)
                            times = gr.Textbox(label="Zaman: olay anları 2, 5.1 · ya da aralık 0-12 (Enter ya da Yeniden üret)", scale=4)
                        with gr.Row():
                            prompt = gr.Textbox(label="İstek", scale=6, lines=1)
                            regen = gr.Button("Yeniden üret", size="sm", scale=1)
                            rm = gr.Button("Sil", size="sm", scale=1)
                        stem = gr.Audio(label="Bu iz tek başına", interactive=False, type="filepath")
                    rows.append(dict(box=box, head=head, gain=gain, pan=pan, dist=dist, mute=mute, solo=solo,
                                     times=times, prompt=prompt, regen=regen, rm=rm, stem=stem))

        slot_outputs = [c for r in rows for c in (r["box"], r["head"], r["gain"], r["pan"], r["dist"], r["mute"],
                                                  r["times"], r["prompt"], r["stem"])]

        def slots():
            ups = []
            for i in range(N_SLOTS):
                s = st.slot(i)
                if s is None:
                    ups += [gr.update(visible=False)] + [gr.update()] * 8
                else:
                    ups += [gr.update(visible=True), s["header"], s["gain"], s["pan"], s["dist"], s["mute"],
                            s["times"], s["prompt"], s["stem"]]
            return ups

        def wrap(fn):
            def run(*a):
                try:
                    out, msg = fn(*a)
                except Exception as e:                                     # noqa: BLE001
                    err(e)
                return [str(out), msg, str(st.job.timeline(st.solo))] + slots()
            return run

        outs = [vout, status, tl] + slot_outputs
        go.click(wrap(lambda v: st.open(v) if v else (_ for _ in ()).throw(ValueError("önce video yükle"))),
                 [vin], outs)
        again.click(wrap(st.regenerate_all), None, outs)
        pack.click(lambda: st.package(), None, pfile)
        add_btn.click(wrap(lambda n, k, p, t, pos: st.add(n, k, p, t, pos)), [a_name, a_kind, a_prompt, a_times, a_pos],
                      outs)
        for i, r in enumerate(rows):
            mix_in = [r["gain"], r["mute"], r["pan"], r["dist"]]
            f_mix = wrap(lambda g, m, p, d, i=i: st.set_mix(i, gain=float(g), mute=bool(m), pan=float(p), dist=float(d)))
            for c in (r["gain"], r["pan"], r["dist"]):
                c.release(f_mix, mix_in, outs)
            r["mute"].input(f_mix, mix_in, outs)
            r["times"].submit(wrap(lambda txt, i=i: st.set_times(i, txt)), [r["times"]], outs)
            r["regen"].click(wrap(lambda p, txt, i=i: st.regenerate(i, p, txt)), [r["prompt"], r["times"]], outs)
            r["prompt"].submit(wrap(lambda p, txt, i=i: st.regenerate(i, p, txt)), [r["prompt"], r["times"]], outs)
            r["solo"].click(wrap(lambda i=i: st.toggle_solo(i)), None, outs)
            r["rm"].click(wrap(lambda i=i: st.remove(i)), None, outs)
    return demo
