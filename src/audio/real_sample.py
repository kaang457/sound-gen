"""Pinned real environmental recording; audio bytes remain outside Git."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np
import soundfile as sf

URL = 'https://librosa.org/data/audio/456440__inspectorj__bird-whistling-robin-single-13.ogg'
SHA256 = '57c2b861d028e25d7c086b48853f20eda1ae9a7a33e1125a1f3bec91539b4208'
ATTRIBUTION = {
    'title': 'Bird Whistling, Robin, Single, 13.wav',
    'creator': 'InspectorJ', 'license': 'CC-BY-4.0',
    'source': 'https://freesound.org/people/InspectorJ/sounds/456440/',
    'mirror': URL, 'change': 'Decode OGG mirror to mono FLOAT WAV; no resampling',
}


def load_robin(root):
    folder = Path(root)/'data'/'raw'
    folder.mkdir(parents=True, exist_ok=True)
    ogg = folder/'robin_inspectorj.ogg'
    if not ogg.exists():
        tmp = ogg.with_suffix('.download')
        try:
            urllib.request.urlretrieve(URL, tmp)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != SHA256:
                raise ValueError('Source hash changed; recheck attribution and source version.')
            tmp.replace(ogg)
        finally:
            tmp.unlink(missing_ok=True)
    if hashlib.sha256(ogg.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Recording hash mismatch.')
    y,sr = sf.read(ogg,dtype='float64',always_2d=True)
    y = y.mean(axis=1)
    if not np.isfinite(y).all(): raise ValueError('Invalid audio.')
    wav = folder/'robin_inspectorj.wav'
    sf.write(wav,y,sr,subtype='FLOAT')
    return y,sr,wav
