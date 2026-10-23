"""Objective checks (none of them replaces listening).

  * CLAP category match: does the output sound like its label? (zero-shot accuracy
    over all labels + similarity to "the sound of <label>")
  * FAD on CLAP embeddings: real held-out clips vs generated, per label
  * diversity: mean pairwise cosine distance between seeds of the same request
  * control check: does turning a knob move the measured quantity the right way?
  * optional baseline: a pretrained text-to-audio diffusion model on the same labels
"""
import numpy as np
import torch

from .audio import SR, rms_db


def label_text(label):
    parts = label.replace("_", " ").split("/")
    return "the sound of " + (f"{parts[0]} on {parts[1]}" if len(parts) > 1 else parts[0])


class Clap:
    def __init__(self, device=None, model_id="laion/clap-htsat-unfused"):
        from transformers import ClapModel, ClapProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = ClapProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def audio(self, wavs, sr=SR):
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(48000, sr)
        wavs = [resample_poly(np.asarray(w, np.float64), 48000 // g, sr // g).astype(np.float32) for w in wavs]
        feats = self.proc.feature_extractor(wavs, sampling_rate=48000, return_tensors="pt")
        feats = {k: v.to(self.device) for k, v in feats.items()}
        e = self.model.get_audio_features(**feats)
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    @torch.no_grad()
    def text(self, texts):
        tok = self.proc.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        e = self.model.get_text_features(**tok)
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def category_match(clap, wavs, labels, all_labels, sr=SR):
    a = clap.audio(wavs, sr)
    t = clap.text([label_text(l) for l in all_labels])
    sims = a @ t.T
    idx = [all_labels.index(l) for l in labels]
    pred = sims.argmax(axis=1)
    return dict(accuracy=float(np.mean(pred == np.array(idx))),
                similarity=float(np.mean(sims[np.arange(len(idx)), idx])),
                predicted=[all_labels[p] for p in pred])


def frechet(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def diversity(emb):
    if len(emb) < 2:
        return float("nan")
    s = emb @ emb.T
    iu = np.triu_indices(len(emb), 1)
    return float(np.mean(1 - s[iu]))


def spectral_centroid(x, sr=SR):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float((X * f).sum() / (X.sum() + 1e-12))


def control_check(gen, label, param, values, seed=0, **kw):
    """Sweep one control; report measured loudness, brightness, length + Spearman rho."""
    from scipy.stats import spearmanr
    from .controls import Controls
    from tqdm.auto import tqdm
    rows = []
    for v in tqdm(values, desc=f"control check: {param}", leave=False):
        out = gen.generate(label, Controls(**{param: v}), seed=seed, **kw)
        a = out["audio"][0]
        rows.append(dict(value=v, rms_db=rms_db(a), centroid_hz=spectral_centroid(a), seconds=len(a) / SR))
    rho = {k: float(spearmanr([r["value"] for r in rows], [r[k] for r in rows]).correlation)
           for k in ("rms_db", "centroid_hz", "seconds")}
    return rows, rho


def diffusion_baseline(labels, seconds=4.0, seed=0, model="stabilityai/stable-audio-open-1.0", steps=100):
    """Zero-shot text-to-audio baseline (gated model: accept its license on HF first)."""
    from diffusers import StableAudioPipeline
    from scipy.signal import resample_poly
    pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
    from tqdm.auto import tqdm
    pipe.set_progress_bar_config(disable=True)
    out = {}
    for label in tqdm(labels, desc="Stable Audio Open", unit="label"):
        g = torch.Generator("cuda").manual_seed(seed)
        a = pipe(label_text(label) + ", isolated sound effect, high quality", negative_prompt="music, speech, low quality",
                 num_inference_steps=steps, audio_end_in_s=seconds, generator=g).audios[0]
        a = a.float().cpu().numpy().mean(axis=0)
        out[label] = resample_poly(a, 1, 2).astype(np.float32)        # 44.1k -> 22.05k
    del pipe
    torch.cuda.empty_cache()
    return out
