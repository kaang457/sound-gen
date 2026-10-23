"""Colab glue: Drive folders and pushing small result files to GitHub.

Secrets (Colab key icon, enable "Notebook access"):
    GITHUB_TOKEN        fine-grained token, only repo kaang457/sound-gen, Contents: read & write
    FREESOUND_API_KEY   optional, https://freesound.org/apiv2/apply
    HF_TOKEN            optional, only for the Stable Audio Open baseline / CLAP downloads
"""
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

REPO = "kaang457/sound-gen"


def secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


def setup(drive=True, root="sound-gen"):
    """Mount Drive and return the project folders (all persistent ones live on Drive)."""
    if drive:
        from google.colab import drive as gdrive
        gdrive.mount("/content/drive")
        base = Path("/content/drive/MyDrive") / root
    else:
        base = Path("/content") / root
    p = SimpleNamespace(
        drive=base,
        raw=Path("/content/raw"),              # big downloads: local disk only
        pack=base / "pack",                    # processed dataset (~1-3 GB)
        custom=base / "custom",                # your own recordings: custom/<label>/*.wav
        runs=base / "runs",                    # checkpoints, logs, samples
        local_pack=Path("/content/pack"),
    )
    for d in (p.raw, p.pack, p.custom, p.runs):
        d.mkdir(parents=True, exist_ok=True)
    hf = secret("HF_TOKEN")
    if hf:
        os.environ["HF_TOKEN"] = hf
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        gpu = "?"
    print(f"GPU: {gpu}\nDrive folder: {base}")
    return p


def local_copy(src, dst):
    """Copy a pack from Drive to the local disk (much faster random reads)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    from tqdm.auto import tqdm
    for f in ["audio.bin", "index.csv", "meta.json"] + [p.name for p in src.glob("clap_*.npy")]:
        s, d = src / f, dst / f
        if s.exists() and (not d.exists() or d.stat().st_size != s.stat().st_size):
            size = s.stat().st_size
            if size < 200e6:
                shutil.copy2(s, d)
                continue
            # big file (the 16 GB audio.bin): copy in chunks with a progress bar; Drive reads ~30-150 MB/s
            tmp = d.with_suffix(d.suffix + ".part")
            with open(s, "rb") as fi, open(tmp, "wb") as fo, tqdm(total=size, unit="B", unit_scale=True,
                                                                   unit_divisor=1024, desc=f"Drive -> {d}") as bar:
                while True:
                    b = fi.read(64 << 20)
                    if not b:
                        break
                    fo.write(b)
                    bar.update(len(b))
            tmp.replace(d)
    return dst


def push_to_github(files, dest="results", message="Add results from Colab", max_mb=20):
    """Copy files into the repo under `dest` and push. Uses the GITHUB_TOKEN secret."""
    token = secret("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Add a GITHUB_TOKEN secret (see README, 'Colab').")
    files = [Path(f) for f in files]
    size = sum(f.stat().st_size for f in files) / 1e6
    if size > max_mb:
        raise ValueError(f"{size:.1f} MB > {max_mb} MB; push fewer/shorter files")
    url = f"https://x-access-token:{token}@github.com/{REPO}.git"
    work = Path("/content/sound-gen-push")

    def git(*args, cwd=work):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr).replace(token, "***"))
        return r.stdout

    # The push folder is only a staging area: reset it to the remote main every time
    # (no merge/rebase, so updates pushed to main in the meantime never conflict).
    if not (work / ".git").exists():
        shutil.rmtree(work, ignore_errors=True)
        git("clone", "--depth", "1", url, str(work), cwd="/content")
    git("remote", "set-url", "origin", url)
    git("config", "user.name", "colab-sound-gen")
    git("config", "user.email", "colab@sound-gen.local")
    for attempt in range(3):
        if any((work / ".git" / d).exists() for d in ("rebase-merge", "rebase-apply")):
            git("rebase", "--abort")
        git("fetch", "--depth", "1", "origin", "main")
        git("reset", "--hard", "FETCH_HEAD")
        target = work / dest
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, target / f.name)
        git("add", dest)
        if not git("status", "--porcelain").strip():
            print("nothing new to push")
            return
        git("commit", "-m", message)
        try:
            git("push", "origin", "HEAD:main")
            break
        except RuntimeError:
            if attempt == 2:
                raise
    print(f"pushed {len(files)} files ({size:.1f} MB) -> https://github.com/{REPO}/tree/main/{dest}")


def health_report(run_dir, packs, push=True):
    """Training health report right after training: build runs/<RUN>/health/report.html and push it.
    Never raises: a failing report or push must not hide a finished training run."""
    from .diagnostics import report
    run = Path(run_dir)
    try:
        findings, path = report(run, packs=[str(p) for p in packs])
    except Exception as e:                       # noqa: BLE001
        print(f"health report skipped: {type(e).__name__}: {e}")
        return None, None
    try:
        from IPython.display import HTML, display
        display(HTML(path.read_text()))
    except Exception:                            # noqa: BLE001
        pass
    if push:
        try:
            push_to_github([path], dest=f"results/{run.name}", message=f"{run.name} health report")
        except Exception as e:                   # noqa: BLE001
            print(f"report saved to {path}, GitHub push failed: {e}")
    return findings, path


def publish(run_dir, packs, refine=None, push=True):
    """Tidy results of a run -> <run>/publish/ (Drive) and GitHub results/<RUN>/ (README, model, bank,
    showcase.html, health.html). Never raises: a failure here must not hide a finished training run."""
    from .publish import publish_run
    run = Path(run_dir)
    try:
        files = publish_run(run, packs, refine=refine)
    except Exception as e:                       # noqa: BLE001
        print(f"publish skipped: {type(e).__name__}: {e}")
        return []
    if push:
        try:
            push_to_github(files, dest=f"results/{run.name}", message=f"{run.name} results", max_mb=190)
        except Exception as e:                   # noqa: BLE001
            print(f"results saved on Drive ({run / 'publish'}), GitHub push failed: {e}")
    return files
