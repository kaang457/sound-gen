"""Pinned CC0 footstep preview, decoded/resampled and cropped for T-FOLEY."""
from pathlib import Path
import hashlib
import urllib.request
import numpy as np
import librosa
import soundfile as sf

SOURCE = {
    'title': 'footsteps.wav', 'creator': 'freefire66', 'license': 'CC0-1.0',
    'source': 'https://freesound.org/people/freefire66/sounds/175954/',
    'download': 'https://cdn.freesound.org/previews/175/175954_2542588-hq.mp3',
    'description': 'Boots on a sidewalk; public compressed preview, not original WAV',
}
SHA256 = '25a963b13fb2b1ab7415525a27326b59cccf3e3ee755d6e8857a38566d268894'


def prepare_footstep(root):
    folder = Path(root) / 'data' / 'raw'
    folder.mkdir(parents=True, exist_ok=True)
    preview = folder / 'footsteps_freefire66_preview.mp3'
    if not preview.exists():
        temp = preview.with_suffix('.download')
        try:
            urllib.request.urlretrieve(SOURCE['download'], temp)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA256:
                raise ValueError('Footstep preview checksum mismatch')
            temp.replace(preview)
        finally:
            temp.unlink(missing_ok=True)
    if hashlib.sha256(preview.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Footstep preview checksum mismatch')
    y, source_sr = sf.read(preview, dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    # Fixed window, selected before model inference; no peak normalization.
    start_s, duration_s, target_sr = 5, 4, 22050
    segment = y[start_s * source_sr:(start_s + duration_s) * source_sr]
    if len(segment) != duration_s * source_sr:
        raise ValueError('Source too short for fixed crop')
    segment = librosa.resample(segment, orig_sr=source_sr, target_sr=target_sr)
    if len(segment) != duration_s * target_sr or not np.isfinite(segment).all():
        raise ValueError('Invalid reference segment')
    path = folder / 'footsteps_freefire66_5s_9s_22050.wav'
    sf.write(path, segment, target_sr, subtype='FLOAT')
    metadata = {**SOURCE, 'preview_sha256': SHA256, 'source_sample_rate': source_sr,
                'crop_start_s': start_s, 'crop_duration_s': duration_s,
                'sample_rate': target_sr, 'peak_normalization': False,
                'changes': 'Decode MP3 preview; mono; fixed 5–9 s crop; resample to 22050 Hz; FLOAT WAV',
                'reference_wav_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return path, metadata
