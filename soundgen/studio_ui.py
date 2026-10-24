"""Gradio front end for the studio (runs in Colab, opens a public share link):

    from soundgen.studio import Studio
    from soundgen.studio_ui import build
    build(Studio("runs/melgen_v1/mel_gen.pt")).queue().launch(share=True)

Left: generate (prompt, optional second prompt + mix, length, prompt strength, seed). "A gunfight" vs "a gunshot"
is the model's job (prompt), there is no rule-based event duplication.
Middle: the 3 variations and 5 blends, pick one. Right: the effects rack, live preview, mono/stereo WAV.
"""
import numpy as np

from .fx import ENVIRONMENTS, Rack

N_SLOTS = 8                                          # 3 variations + 5 blends (soundgen/mm/generate.py PLAN)

CSS = """
.gradio-container {max-width: 1320px !important; background: #16171a !important;}
#title h1 {font-size: 22px; margin: 4px 0 0; letter-spacing: .5px}
#title p {color: #9a9ca3; margin: 2px 0 8px}
.panel {background: #1f2125 !important; border: 1px solid #2e3036 !important; border-radius: 10px !important;
        padding: 10px !important}
.panel h3 {font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; color: #f2a33a; margin: 0 0 6px}
.slot {border: 1px solid #2e3036 !important; border-radius: 8px !important; background: #191a1d !important}
#go {background: #f2a33a !important; color: #111 !important; font-weight: 700}
"""

ENV_LABELS = {"dry": "Kuru", "room": "Oda", "hall": "Salon", "cave": "Mağara", "outdoor": "Açık alan"}


def _to_gradio(x, sr):
    """float [-1,1] mono [T] or stereo [2,T] -> (sr, int16 [T] / [T,2]) for gr.Audio."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.T
    return sr, (np.clip(x, -1, 1) * 32767).astype(np.int16)


def build(studio, title="Ses Stüdyosu"):
    import gradio as gr

    theme = gr.themes.Base(primary_hue="orange", neutral_hue="zinc").set(
        body_background_fill="#16171a", block_background_fill="#1f2125", body_text_color="#e8e8ea",
        block_label_text_color="#9a9ca3", input_background_fill="#26282d", border_color_primary="#2e3036")

    with gr.Blocks(title=title, theme=theme, css=CSS) as demo:
        sounds = gr.State([])
        gr.Markdown(f"# {title}\nİstek yaz → sıfırdan 3 varyasyon + aralarında 5 karışım → düzenle → indir.",
                    elem_id="title")
        with gr.Row(equal_height=False):
            # ---------------------------------------------------------------- generate
            with gr.Column(scale=3, elem_classes="panel"):
                gr.Markdown("### Üret")
                prompt = gr.Textbox(label="İstek", value="laser gunshot", placeholder="ör. spaceship engine hum")
                prompt_b = gr.Textbox(label="İkinci istek (opsiyonel)", placeholder="ör. zombie growl")
                mix_b = gr.Slider(0, 1, 0.0, step=0.05, label="İkinci isteğin payı")
                seconds = gr.Slider(1.0, 8.0, 4.0, step=0.5, label="Uzunluk (s)")
                guidance = gr.Slider(1.0, 8.0, 4.5, step=0.5, label="İstek gücü")
                seed = gr.Number(0, precision=0, label="Tohum")
                go = gr.Button("Üret", elem_id="go")
                status = gr.Markdown("")
            # ---------------------------------------------------------------- results
            with gr.Column(scale=4, elem_classes="panel"):
                gr.Markdown("### Sesler")
                slots = []
                for row in range(0, N_SLOTS, 2):
                    with gr.Row():
                        for k in range(row, min(row + 2, N_SLOTS)):
                            slots.append(gr.Audio(label=f"#{k + 1}", interactive=False, visible=False,
                                                  elem_classes="slot", show_download_button=True))
                pick = gr.Radio([], label="Düzenlenecek ses")
            # ---------------------------------------------------------------- effects rack
            with gr.Column(scale=3, elem_classes="panel"):
                gr.Markdown("### İşle")
                intensity = gr.Slider(-24, 12, 0, step=0.5, label="Şiddet (dB)")
                dist = gr.Slider(0, 1, 0, step=0.01, label="Uzaklık (yakın → uzak)")
                env = gr.Radio([(v, k) for k, v in ENV_LABELS.items()], value="dry", label="Ortam")
                wet = gr.Slider(0, 1, 0.3, step=0.01, label="Yankı miktarı")
                pan = gr.Slider(-1, 1, 0, step=0.05, label="Stereo konum (sol ↔ sağ)")
                stretch = gr.Slider(0.5, 2.0, 1.0, step=0.05, label="Süre çarpanı (perde korunur)")
                with gr.Row():
                    trim_a = gr.Number(0.0, label="Baştan kırp (s)")
                    trim_b = gr.Number(0.0, label="Sondan kırp (s)")
                with gr.Row():
                    fade_in = gr.Number(5, label="Fade-in (ms)")
                    fade_out = gr.Number(30, label="Fade-out (ms)")
                preview = gr.Audio(label="Önizleme (stereo)", interactive=False)
                with gr.Row():
                    f_mono = gr.File(label="Mono WAV")
                    f_stereo = gr.File(label="Stereo WAV")
                all_btn = gr.Button("Hepsini bu ayarlarla indir (zip)")
                f_all = gr.File(label="Zip")

        rack_inputs = [intensity, dist, env, wet, pan, stretch, trim_a, trim_b, fade_in, fade_out]

        def make_rack(i, d, e, w, p, s, ta, tb, fi, fo):
            return Rack(intensity_db=float(i), distance=float(d), environment=e,
                        wet=float(w) if e != "dry" else 0.0, pan=float(p), stretch=float(s),
                        trim_start=float(ta or 0), trim_end=float(tb or 0),
                        fade_in_ms=float(fi or 0), fade_out_ms=float(fo or 0))

        def on_env(e):
            p = ENVIRONMENTS.get(e)
            return gr.update(value=p["wet"] if p else 0.0, interactive=p is not None)

        def on_generate(pr, pb, mb, sec, gd, sd):
            if not pr or not pr.strip():
                raise gr.Error("Bir istek yaz")
            out = studio.generate(pr, pb, mb, seconds=sec, seed=int(sd or 0), guidance=gd)
            ups = []
            for k in range(N_SLOTS):
                if k < len(out):
                    s = out[k]
                    ups.append(gr.update(value=_to_gradio(s["audio"], s["sr"]), visible=True,
                                         label=f"{s['name']} · {s['kind']}"))
                else:
                    ups.append(gr.update(value=None, visible=False))
            names = [s["name"] for s in out]
            msg = f"{len(out)} ses üretildi: “{pr}”" + (f" + %{int(mb * 100)} “{pb}”" if pb and mb > 0 else "")
            return [out, gr.update(choices=names, value=names[0]), msg] + ups

        def on_rack(snds, name, *vals):
            if not snds or not name:
                return None, None, None
            s = next(x for x in snds if x["name"] == name)
            fm, fs, stereo, _ = studio.export(s, make_rack(*vals))
            return _to_gradio(stereo, s["sr"]), str(fm), str(fs)

        def on_all(snds, *vals):
            if not snds:
                raise gr.Error("Önce üret")
            return str(studio.export_all(snds, make_rack(*vals)))

        go.click(on_generate, [prompt, prompt_b, mix_b, seconds, guidance, seed],
                 [sounds, pick, status] + slots).then(on_rack, [sounds, pick] + rack_inputs,
                                                     [preview, f_mono, f_stereo])
        args, outs = [sounds, pick] + rack_inputs, [preview, f_mono, f_stereo]
        env.change(on_env, env, wet).then(on_rack, args, outs)        # preset wet first, then render
        pick.change(on_rack, args, outs)
        for c in rack_inputs:
            if c is env:
                continue
            (c.release if isinstance(c, gr.Slider) else c.change)(on_rack, args, outs)
        all_btn.click(on_all, [sounds] + rack_inputs, f_all)
    return demo
