"""Pack -> training features, computed once and cached (the training loop then never touches audio).

    extract(packs, out_dir, raw_dir)  ->  out_dir/
        index.tsv        one row per clip: id, label, split, caption index, caption
        captions.json    the unique captions
        mean.npy, std.npy   VAE posterior per clip, [N, 345, 40] fp16 (8 s at 44.1 kHz)
        text.npy         CLIP text tokens per UNIQUE caption, [U, 77, 1024] fp16
        meta.json        sizes, versions

Audio: each pack chunk (<= 6 s, 48 kHz) -> 44.1 kHz -> peak 0.95 -> zero-padded to 353,280 samples (8.01 s, a
multiple of hop 512 x VAE downsampling 2), as in MMAudio's own extraction.

Captions: FSD50K clips get their (multi-)labels from the FSD50K ground truth plus the Freesound title when it is
informative ("gunshot and gunfire, explosion; pistol shot in alley"); other sources get their category name.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

NUM_SAMPLES = 353_280
SR = 44_100
CLIP_ID = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
TOKENIZER_ID = "ViT-H-14-378-quickgelu"


# ------------------------------------------------------------------ captions
def _words(s):
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()


def _title_ok(t):
    t = re.sub(r"\.(wav|mp3|flac|aif+|ogg)$", "", t or "", flags=re.I)
    t = re.sub(r"[_\-]+", " ", t)
    words = [w for w in re.findall(r"[A-Za-z]+", t) if len(w) >= 3]
    return " ".join(t.split()).lower() if len(words) >= 2 else ""


def fsd50k_captions(raw_dir):
    """{fname: caption} for FSD50K dev + eval from the small metadata downloads (~10 MB)."""
    from ..data import fsd50k_index
    out = {}
    for split in ("dev", "eval"):
        try:
            rows, clips = fsd50k_index(raw_dir, split)
        except Exception as e:                       # noqa: BLE001
            print(f"FSD50K {split} metadata unavailable ({type(e).__name__}); using category names")
            continue
        for r in rows:
            labels = ", ".join(_words(l) for l in r["labels"].split(","))
            title = _title_ok(clips.get(str(r["fname"]), {}).get("title", ""))
            out[str(r["fname"])] = labels + (f"; {title}" if title and title not in labels else "")
    return out


def caption_for(row, fsd):
    if row.get("source") == "fsd50k" and str(row.get("src_id")) in fsd:
        return fsd[str(row["src_id"])]
    lab = row["label"]
    return "sound effect" if lab == "other" else _words(lab.split("/")[0])


# ------------------------------------------------------------------ models
def text_encoder(device):
    """MMAudio's text encoder: DFN5B CLIP ViT-H/14, last hidden states (77 x 1024), L2-normalised per token."""
    import open_clip
    import torch.nn.functional as F
    model = open_clip.create_model_from_pretrained(CLIP_ID, return_transform=False).eval().to(device)

    def encode_text(self, text, normalize=False):          # same patch as MMAudio's features_utils.patch_clip
        cast = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast) + self.positional_embedding.to(cast)
        x = self.ln_final(self.transformer(x, attn_mask=self.attn_mask))
        return F.normalize(x, dim=-1) if normalize else x

    model.encode_text = encode_text.__get__(model)
    tok = open_clip.get_tokenizer(TOKENIZER_ID)

    @torch.inference_mode()
    def encode(texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            out.append(model.encode_text(tok(texts[i: i + batch]).to(device), normalize=True).float().cpu())
        return torch.cat(out)
    return encode


class Codec:
    """mel converter + MMAudio's 44.1 kHz VAE (+ BigVGAN-v2 vocoder for decoding)."""

    def __init__(self, vae_ckpt, device, need_encoder=True):
        from mmaudio.ext.autoencoder import AutoEncoderModule
        from mmaudio.ext.mel_converter import get_mel_converter
        self.device = device
        self.mel = get_mel_converter("44k").to(device)
        self.tod = AutoEncoderModule(vae_ckpt_path=str(vae_ckpt), mode="44k",
                                     need_vae_encoder=need_encoder).eval().to(device)

    @torch.inference_mode()
    def encode(self, wav):
        """[B, 353280] at 44.1 kHz -> (mean, std) [B, 345, 40]"""
        dist = self.tod.encode(self.mel(wav.to(self.device)))
        return dist.mean.transpose(1, 2), dist.std.transpose(1, 2)

    @torch.inference_mode()
    def decode(self, latent):
        """[B, N, 40] (unnormalised) -> waveform [B, N * 1024]"""
        mel = self.tod.decode(latent.transpose(1, 2))
        return self.tod.vocode(mel).squeeze(1)


# ------------------------------------------------------------------ extraction
class _Clips(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.audio = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from ..data import pack_sample_rate, resample
        r = self.rows[i]
        if r["pack"] not in self.audio:
            self.audio[r["pack"]] = (np.memmap(Path(r["pack"]) / "audio.bin", dtype="<i2", mode="r"),
                                     pack_sample_rate(r["pack"]))
        mm, src = self.audio[r["pack"]]
        a = np.asarray(mm[r["offset"]: r["offset"] + r["length"]], np.float32) / 32768
        a = resample(a, src, SR)[:NUM_SAMPLES]
        peak = np.abs(a).max()
        if peak > 1e-6:
            a = a / peak * 0.95
        out = np.zeros(NUM_SAMPLES, np.float32)
        out[: len(a)] = a
        return torch.from_numpy(out), i


def extract(packs, out_dir, raw_dir, vae_ckpt, device=None, batch=32, num_workers=4, codec=None, encode_text=None):
    """See module docstring. Resumable: finished clip batches are kept in mean/std (row k done when done[k])."""
    from tqdm.auto import tqdm
    from ..data import read_index
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for p in packs for r in read_index(p)]
    fsd = fsd50k_captions(raw_dir) if any(r.get("source") == "fsd50k" for r in rows) else {}
    caps = [caption_for(r, fsd) for r in rows]
    uniq = sorted(set(caps))
    cidx = {c: k for k, c in enumerate(uniq)}
    n = len(rows)
    with open(out / "index.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "label", "split", "source", "cap", "caption"])
        for r, c in zip(rows, caps):
            w.writerow([r["id"], r["label"], r["split"], r.get("source", ""), cidx[c], c])
    (out / "captions.json").write_text(json.dumps(uniq, ensure_ascii=False))
    print(f"{n} clips, {len(uniq)} unique captions; e.g. {uniq[:3]}")

    # text tokens for the unique captions
    if not (out / "text.npy").exists():
        enc = encode_text or text_encoder(device)
        np.save(out / "text.npy", enc(uniq).numpy().astype(np.float16))
        del enc
    # audio latents
    open_mm = lambda name: (np.lib.format.open_memmap(out / name, mode="r+") if (out / name).exists() else
                            np.lib.format.open_memmap(out / name, mode="w+", dtype=np.float16, shape=(n, 345, 40)))
    mean, std = open_mm("mean.npy"), open_mm("std.npy")
    done_f = out / "done.npy"
    done = np.load(done_f) if done_f.exists() else np.zeros(n, bool)
    todo = [k for k in range(n) if not done[k]]
    if todo:
        codec = codec or Codec(vae_ckpt, device)
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(_Clips(rows), todo), batch_size=batch,
                                         num_workers=num_workers)
        for b, (wav, idx) in enumerate(tqdm(dl, desc="VAE latents", unit="batch")):
            m, s = codec.encode(wav)
            idx = idx.numpy()
            mean[idx] = m.float().cpu().numpy().astype(np.float16)
            std[idx] = s.float().cpu().numpy().astype(np.float16)
            done[idx] = True
            if b % 50 == 49:
                mean.flush(), std.flush(), np.save(done_f, done)
        mean.flush(), std.flush(), np.save(done_f, done)
    meta = dict(n=n, captions=len(uniq), seq_len=345, latent_dim=40, sr=SR, num_samples=NUM_SAMPLES,
                clip=CLIP_ID, packs=[str(p) for p in packs])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return out


def load(features_dir):
    """-> dict(rows, mean, std (memmaps), text [U,77,1024], cap_idx [N], meta)"""
    d = Path(features_dir)
    with open(d / "index.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return dict(rows=rows, mean=np.load(d / "mean.npy", mmap_mode="r"), std=np.load(d / "std.npy", mmap_mode="r"),
                text=np.load(d / "text.npy"), cap_idx=np.array([int(r["cap"]) for r in rows]),
                meta=json.loads((d / "meta.json").read_text()))
