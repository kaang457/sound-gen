"""Single-pair temporal diagnostics; no timbre/category similarity claim."""
from pathlib import Path
import csv
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import ellip, filtfilt, find_peaks
import matplotlib.pyplot as plt


def compare_pair(root, stem, source_metadata):
    root = Path(root)
    folder = root / 'results/audio/tfoley'
    generated, sr = sf.read(folder / (stem + '.wav'), dtype='float64')
    reference, ref_sr = sf.read(folder / (stem + '_reference.wav'), dtype='float64')
    if sr != ref_sr or generated.shape != reference.shape or generated.ndim != 1:
        raise ValueError('Expected equal-length mono audio at the same sample rate')
    if not all(np.isfinite(z).all() for z in [generated, reference]):
        raise ValueError('Nonfinite audio')
    rms = [librosa.feature.rms(y=z.astype(np.float32), frame_length=512, hop_length=128)[0]
           for z in [reference, generated]]
    b, a = ellip(4, 0.01, 120, 0.125)
    smoothed = [filtfilt(b, a, z, method='gust') for z in rms]
    times = librosa.frames_to_time(np.arange(len(rms[0])), sr=sr, hop_length=128)
    peaks = [find_peaks(z, distance=round(0.25 * sr / 128),
                        prominence=max(1e-8, float(np.max(z)) * 0.15))[0]
             for z in smoothed]
    reference_peaks, generated_peaks = [times[p] for p in peaks]
    # Nearest peak per reference; not one-to-one, not an onset detector.
    offsets = ([float(generated_peaks[np.argmin(abs(generated_peaks - t))] - t)
                for t in reference_peaks] if len(generated_peaks) else [])
    metrics = {
        'source': source_metadata, 'output_stem': stem, 'sample_rate': sr,
        'duration_s': len(generated) / sr,
        'smoothed_rms_mae': float(np.mean(abs(smoothed[0] - smoothed[1]))),
        'smoothed_rms_pearson': (float(np.corrcoef(*smoothed)[0, 1])
                                 if all(np.std(z) > 0 for z in smoothed) else None),
        'reference_global_rms': float(np.sqrt(np.mean(reference ** 2))),
        'generated_global_rms': float(np.sqrt(np.mean(generated ** 2))),
        'generated_peak': float(np.max(abs(generated))),
        'generated_fraction_abs_above_1': float(np.mean(abs(generated) > 1)),
        'reference_rms_peak_times_s': reference_peaks.tolist(),
        'generated_rms_peak_times_s': generated_peaks.tolist(),
        'nearest_peak_offsets_s': offsets,
        'peak_diagnostic': 'Relative prominence 15%, distance 250 ms; RMS peaks, not onsets; nearest matching may reuse peaks',
        'scope': 'Single-pair engineering diagnostics; no benchmark, perceptual similarity or category verification',
    }
    exp = root / 'experiments/baselines/tfoley/real_footstep'
    table = root / 'results/tables/tfoley_real_footstep'
    figures = root / 'results/figures/tfoley_real_footstep'
    for path in [exp, table, figures]: path.mkdir(parents=True, exist_ok=True)
    (exp / (stem + '_comparison.json')).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    with (table / (stem + '_rms.csv')).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'reference_rms', 'generated_rms', 'reference_smoothed_rms', 'generated_smoothed_rms'])
        w.writerows(zip(times, *rms, *smoothed))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    for values, label in zip(rms, ['Reference RMS', 'Generated RMS']):
        axes[0].plot(times, values, label=label)
    axes[0].set(ylabel='Unweighted RMS'); axes[0].legend()
    for values, label in zip(smoothed, ['Reference condition', 'Generated smoothed RMS']):
        axes[1].plot(times, values, label=label)
    axes[1].set(xlabel='Time (s)', ylabel='Smoothed RMS'); axes[1].legend()
    fig.suptitle('Real Footstep reference vs T-FOLEY (single sample)')
    fig.savefig(figures / (stem + '.svg'))
    return metrics, fig
