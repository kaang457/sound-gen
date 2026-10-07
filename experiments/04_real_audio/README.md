# Gerçek çevresel ses analizi

## Veri kaynağı

“Bird Whistling, Robin, Single, 13.wav”, InspectorJ, [Freesound 456440](https://freesound.org/people/InspectorJ/sounds/456440/), CC-BY-4.0. Lisans ve kayıt sahibi 7 Ekim 2026'da doğrulandı.

Analiz, orijinal 44.1 kHz stereo kaydın librosa tarafından sağlanan 22.05 kHz mono OGG kopyasını kullanır. Bu dosya bir gerçek çevresel kayıt türevidir; sentetik değildir. OGG çözülüp FLOAT WAV'a yazılır; bu adımda yeniden örnekleme veya tepe normalizasyonu yapılmaz. İlk notebook çalıştırması kayıt indirmesini ve SHA-256 kontrolünü içerir. Audio Git'e eklenmez.

## Kod ve ölçümler

Notebook: [04_real_audio_analysis](../../notebooks/04_real_audio_analysis.ipynb) / [05_pitch_periodicity](../../notebooks/05_pitch_periodicity.ipynb).
Ortak kod: [features.py](../../src/audio/features.py). Kaynak indirme ve atıf: [real_sample.py](../../src/audio/real_sample.py).

| Ölçüm | Değer |
|---|---:|
| sr | 22050 |
| duration_s | 2.698639455782313 |
| frames | 233 |
| active_frames | 175 |
| global_rms | 0.070138618723432 |
| mean_centroid_active_hz | 5346.095488675792 |

## Analiz ve sınırlar

Frame uzunluğu 2048, hop 256; center True ve sıfır padding. Eşik −50 dBFS, kalibre edilmiş loudness veya onset sınırı değildir. RMS dikdörtgen frame'de, centroid Hann STFT büyüklüğünden hesaplanır. Düşük seviyede centroid maskelenir.

## Çıktılar ve doğrulama

[Frame CSV](../../results/tables/04_real_audio/frame_features.csv), [run metadata](run.json) ve grafikler `results/figures/04_real_audio/` altında. Tanımsız özellikler CSV'de boş kaydedilir. Tekrar çalıştırma aynı çıktı dosyalarını günceller.

Kod hücreleri ve notebook şemaları doğrulandı. Koşu Linux/Python 3.12.14 ortamındadır; paket sürümleri run.json'da. Windows/Python 3.11 koşusu henüz doğrulanmadı. Kalite, model controllability veya disentanglement değerlendirmesi yapılmadı.
