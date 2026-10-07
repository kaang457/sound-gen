"""Audio-reference generation diagnostics; no semantic similarity metric."""
from pathlib import Path
import csv
import json
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt


def save_diagnostics(root, stem):
    root = Path(root)
    folder = root / 'results/audio/audioldm'
    table = root / 'results/tables/audioldm'
    figures = root / 'results/figures/audioldm'
    for path in [table, figures]: path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    summary = []
    for ax, name, suffix in zip(axes, ['reference', 'generated'], ['_reference', '']):
        y, sr = sf.read(folder / (stem + suffix + '.wav'), dtype='float64')
        if sr != 16000 or y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError('Expected finite mono 16k audio')
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        db = librosa.amplitude_to_db(S, ref=1.0, top_db=None)
        ax.imshow(db, origin='lower', aspect='auto', extent=[0, len(y)/sr, 0, sr/2],
                  vmin=-80, vmax=0, cmap='magma')
        ax.set(title=name, xlabel='Time (s)', ylabel='Frequency (Hz)')
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        active = rms >= 10 ** (-50/20)
        summary.append({'signal':name, 'duration_s':len(y)/sr,
                        'rms':float(np.sqrt(np.mean(y*y))), 'peak':float(np.max(abs(y))),
                        'mean_centroid_active_hz':float(np.mean(centroid[active])) if np.any(active) else None,
                        'active_frames':int(active.sum()), 'frames':len(rms),
                        'frame_length':1024, 'hop_length':256, 'rms_threshold_dbfs':-50,
                        'role':'signal diagnostics only; not semantic similarity or control success'})
        with (table / (stem + '_' + name + '_frames.csv')).open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['time_s', 'rms', 'centroid_active_hz'])
            w.writerows((i*256/sr, float(v), float(centroid[i]) if active[i] else '') for i, v in enumerate(rms))
    fig.suptitle('AudioLDM audio-reference conditioning: Robin recording vs generated sample')
    fig.savefig(figures / (stem + '.svg'))
    (table / (stem + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, fig
