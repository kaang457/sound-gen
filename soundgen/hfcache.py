"""Where Hugging Face weights live in Colab, and how a broken download repairs itself.

Rules:
  * The cache is on the LOCAL disk (/content/hf_cache). Google Drive's FUSE mount corrupts the hub's chunked
    (xet) downloads and sometimes returns empty files: "header too large", "JSONDecodeError: Expecting value".
  * Drive keeps one tar of the cache (symlinks preserved): a new session or an OFFLINE run restores it in ~1 min
    instead of downloading again.
  * Works even when huggingface_hub / transformers were already imported (their cache constants are patched at
    runtime), so no "restart the session" is needed, and old notebook cells that pointed HF_HOME at Drive are
    corrected the moment a model is loaded (ensure_local).
  * A load that fails on a corrupted file deletes that repo from the cache and downloads it once more (load()).

    from soundgen import hfcache
    tar = hfcache.setup(P.drive, offline=False)
    model = hfcache.load(lambda: SomeModel.from_pretrained(...), repos=["org/name"])
    hfcache.backup(tar)
"""
import os
import subprocess
import sys
from pathlib import Path

LOCAL = Path("/content/hf_cache") if Path("/content").exists() else Path.home() / ".cache" / "soundgen_hf"
TAR_NAME = "hf_cache.tar"
_state = dict(tar=None)


def _patch(local, offline):
    """Point every cache path of huggingface_hub (and transformers' copy) at `local`, also if already imported."""
    local = Path(local)
    env = dict(HF_HOME=local, HF_HUB_CACHE=local / "hub", HF_XET_CACHE=local / "xet", HF_ASSETS_CACHE=local / "assets")
    for k, v in env.items():
        os.environ[k] = str(v)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    c = sys.modules.get("huggingface_hub.constants")
    if c is not None:
        for k, v in env.items():
            if hasattr(c, k):
                setattr(c, k, str(v))
        if hasattr(c, "HF_HUB_OFFLINE") and offline:
            c.HF_HUB_OFFLINE = True
    for name in ("transformers.utils.hub", "transformers.utils"):
        m = sys.modules.get(name)
        if m is not None:
            for k in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE", "PYTORCH_TRANSFORMERS_CACHE"):
                if hasattr(m, k):
                    setattr(m, k, str(local / "hub"))
            if hasattr(m, "HF_MODULES_CACHE"):
                m.HF_MODULES_CACHE = str(local / "modules")


def current():
    c = sys.modules.get("huggingface_hub.constants")
    return getattr(c, "HF_HUB_CACHE", None) if c else os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")


def setup(backup_dir=None, offline=False, local=LOCAL):
    """Local cache + optional Drive backup tar (restored if the local cache is empty). -> tar path or None."""
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    _patch(local, offline)
    tar = Path(backup_dir) / TAR_NAME if backup_dir else None
    _state["tar"] = tar
    if tar and tar.exists() and not any((local / "hub").glob("models--*")):
        print(f"model önbelleği Drive yedeğinden açılıyor ({tar.stat().st_size / 2**30:.1f} GB)…")
        subprocess.run(["tar", "-xf", str(tar), "-C", str(local)], check=True)
    elif offline and not any((local / "hub").glob("models--*")):
        raise RuntimeError("OFFLINE: yerelde ve Drive'da model önbelleği yok; önce bir kez OFFLINE = False ile çalıştır")
    return tar


def ensure_local():
    """Called by every model loader: if the cache still points at Drive (old notebook cell), move it local."""
    cur = str(current() or "")
    if cur.startswith("/content/drive") or not current():
        if cur:
            print(f"model önbelleği Drive'daydı ({cur}); yerel diske alınıyor")
        setup(_state["tar"].parent if _state["tar"] else None, offline=os.environ.get("HF_HUB_OFFLINE") == "1")


def _drop(repo):
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(current())
        revs = [r.commit_hash for rr in info.repos if rr.repo_id == repo for r in rr.revisions]
        if revs:
            info.delete_revisions(*revs).execute()
            print(f"bozuk önbellek silindi: {repo}")
    except Exception as e:                                             # noqa: BLE001
        print(f"önbellek temizlenemedi ({repo}): {type(e).__name__}: {e}")


def load(fn, repos):
    """fn() loads a model. If it fails for any reason other than access / missing repo / no internet,
    the cached files of `repos` are deleted and fn() runs once more (fresh download)."""
    ensure_local()
    try:
        return fn()
    except Exception as e:                                             # noqa: BLE001
        name = type(e).__name__
        if name in ("GatedRepoError", "RepositoryNotFoundError", "LocalEntryNotFoundError", "OfflineModeIsEnabled",
                    "OutOfMemoryError", "ImportError", "ModuleNotFoundError") or "CUDA out of memory" in str(e):
            raise
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError(f"OFFLINE yüklemede hata ({name}: {e}); önbellek bozuk olabilir: "
                               "bir kez OFFLINE = False ile çalıştır") from e
        print(f"yükleme hatası ({name}: {str(e)[:160]}); önbellekteki dosyalar bozuk sayılıp yeniden indiriliyor")
        for r in repos:
            _drop(r)
        return fn()


def backup(tar=None, local=LOCAL):
    """Write / refresh the Drive tar when the local cache has grown (a new model was downloaded)."""
    tar = Path(tar) if tar else _state["tar"]
    hub = Path(local) / "hub"
    if not tar or not hub.exists():
        return
    size = sum(f.stat().st_size for f in hub.rglob("*") if f.is_file() and not f.is_symlink())
    if tar.exists() and tar.stat().st_size >= 0.98 * size:
        return
    tar.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar.with_name(tar.name + ".part")
    print(f"model önbelleği Drive'a yedekleniyor ({size / 2**30:.1f} GB, bir kez; sonraki oturumlar buradan açılır)…")
    subprocess.run(["tar", "-cf", str(tmp), "-C", str(local), "hub"], check=True)
    tmp.replace(tar)
