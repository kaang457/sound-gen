"""Make MMAudio importable without pip-installing it (its pinned numpy<2.1 would break Colab's torch stack).

    root = ensure_mmaudio()          # clones the pinned commit, adds it to sys.path
    w = ext_weights()                # {"vae": .../v1-44.pth, "empty": .../empty_string.pth}

Python deps that MMAudio needs for what we use (network, VAE, BigVGAN-v2, CLIP text): einops, open_clip_torch,
torchdiffeq, librosa, huggingface_hub, omegaconf; Colab has most of them, `pip_deps()` installs the rest without touching
torch / numpy.
"""
import subprocess
import sys
from pathlib import Path

MMAUDIO_GIT = "https://github.com/hkchengrex/MMAudio"
MMAUDIO_COMMIT = "974010a"                      # tested with this commit (Oct 2026)
RELEASE = "https://github.com/hkchengrex/MMAudio/releases/download/v0.1"
WEIGHTS = {"vae": ("v1-44.pth", "fab020275fa44c6589820ce025191600"),   # 44.1 kHz mel-VAE, 1.2 GB
           "empty": ("empty_string.pth", None)}                       # CLIP features of "" (CFG null text)
CACHE = Path.home() / ".cache" / "soundgen"


def pip_deps():
    """Install the small pure-python deps MMAudio needs, without upgrading/downgrading torch or numpy."""
    need = []
    for mod, pkg in [("einops", "einops"), ("open_clip", "open_clip_torch"), ("torchdiffeq", "torchdiffeq"),
                     ("librosa", "librosa"), ("huggingface_hub", "huggingface_hub"),
                     ("omegaconf", "omegaconf")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
    return need


def ensure_mmaudio(root=None):
    root = Path(root or CACHE / "MMAudio")
    if not (root / "mmaudio" / "model" / "networks.py").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", MMAUDIO_GIT, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", MMAUDIO_COMMIT], check=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ext_weights(dest=None):
    import hashlib
    import urllib.request
    dest = Path(dest or CACHE / "ext_weights")
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (name, md5) in WEIGHTS.items():
        f = dest / name
        if not f.exists():
            print(f"downloading {name} ...")
            tmp = f.with_suffix(".part")
            urllib.request.urlretrieve(f"{RELEASE}/{name}", tmp)
            tmp.rename(f)
        if md5:
            h = hashlib.md5()
            with open(f, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 24), b""):
                    h.update(b)
            if h.hexdigest() != md5:
                f.unlink()
                raise IOError(f"{name}: md5 mismatch, deleted; run again")
        out[key] = f
    return out
