"""Frame features with explicit definitions; no model-specific conditioning."""
import numpy as np
import librosa


def frame_features(y, sr, frame_length=2048, hop_length=256, threshold_db=-50):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError('Expected finite, non-empty mono audio.')
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length,
                             center=True, pad_mode='constant', dtype=np.float64)[0]
    mag = np.abs(librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length,
                            center=True, pad_mode='constant', window='hann'))
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    db = 20*np.log10(np.maximum(rms, 1e-10))
    active = db > threshold_db
    assert len(rms) == len(centroid)
    return dict(time_s=np.arange(len(rms))*hop_length/sr, rms=rms,
                rms_dbfs=db, active=active, centroid_hz=np.where(active, centroid, np.nan))


def pitch_features(y, sr, frame_length=2048, hop_length=256, fmin=150,
                   fmax=8000, threshold_db=-50, probability_threshold=0.8):
    """pYIN voicing probability plus normalized autocorrelation at estimated F0.

    Voicing probability and ACF score are algorithm-dependent proxies, not CREPE
    periodicity, calibrated confidence, or a pitch ground truth.
    """
    if not 0 < fmin < fmax < sr/2:
        raise ValueError('Require 0 < fmin < fmax < Nyquist.')
    base = frame_features(y, sr, frame_length, hop_length, threshold_db)
    f0, flag, probability = librosa.pyin(
        y=np.asarray(y,dtype=np.float64), sr=sr, fmin=fmin, fmax=fmax,
        frame_length=frame_length, hop_length=hop_length,
        center=True, pad_mode='constant', fill_na=np.nan)
    reliable = flag & (probability >= probability_threshold) & base['active'] & np.isfinite(f0)
    padded = np.pad(np.asarray(y,dtype=np.float64), frame_length//2)
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    score = np.full(len(f0), np.nan)
    for i in np.flatnonzero(reliable):
        x = frames[:,i] - frames[:,i].mean()
        lag = int(round(sr/f0[i]))
        if 0 < lag < frame_length:
            a,b = x[:-lag],x[lag:]
            denom = np.sqrt(np.dot(a,a)*np.dot(b,b))
            if denom > 0: score[i] = np.clip(np.dot(a,b)/denom,-1,1)
    assert len(f0) == len(base['rms'])
    base.update(f0_raw_hz=f0, f0_hz=np.where(reliable,f0,np.nan),
                voiced_flag=flag, voiced_probability=probability,
                reliable_pitch=reliable, acf_periodicity_proxy=score)
    return base
