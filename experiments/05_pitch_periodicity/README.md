# Pitch ve periodicity incelemesi

## Veri kaynağı

“Bird Whistling, Robin, Single, 13.wav”, InspectorJ, [Freesound 456440](https://freesound.org/people/InspectorJ/sounds/456440/), CC-BY-4.0. Lisans ve kayıt sahibi 7 Ekim 2026'da doğrulandı.

Analiz, orijinal 44.1 kHz stereo kaydın librosa tarafından sağlanan 22.05 kHz mono OGG kopyasını kullanır. Bu dosya bir gerçek çevresel kayıt türevidir; sentetik değildir. OGG çözülüp FLOAT WAV'a yazılır; bu adımda yeniden örnekleme veya tepe normalizasyonu yapılmaz. İlk notebook çalıştırması kayıt indirmesini ve SHA-256 kontrolünü içerir. Audio Git'e eklenmez.

## Kod ve ölçümler

Notebook: [04_real_audio_analysis](../../notebooks/04_real_audio_analysis.ipynb) / [05_pitch_periodicity](../../notebooks/05_pitch_periodicity.ipynb).
Ortak kod: [features.py](../../src/audio/features.py). Kaynak indirme ve atıf: [real_sample.py](../../src/audio/real_sample.py).

| Ölçüm | Değer |
|---|---:|
| frames | 233 |
| reliable_pitch_frames | 60 |
| reliable_pitch_fraction | 0.2575107296137339 |
| median_f0_reliable_hz | 5561.75588119227 |
| mean_voicing_probability | 0.39250468330235677 |

## Analiz ve sınırlar

Frame uzunluğu 2048, hop 256; center True ve sıfır padding. Eşik −50 dBFS, kalibre edilmiş loudness veya onset sınırı değildir. RMS dikdörtgen frame'de, centroid Hann STFT büyüklüğünden hesaplanır. Düşük seviyede centroid maskelenir.

F0 için pYIN, 150–8000 Hz aralığı ve 0.8 voicing probability eşiği kullanılır. Algoritmanın voiced flag'i, probability eşiği ve RMS maskesi birlikte uygulanır. ACF periodicity proxy, tahmin edilmiş F0 periyodundaki normalize korelasyondur; CREPE periodicity değildir ve F0'dan bağımsız kanıt değildir.

Kuş kaydında 233 frame'in 60'ı maskeyi geçti; yalnızca bu frame'lerin F0 medyanı raporlandı. Bu bir otomatik tahmindir; elle etiketli F0 doğruluğu ölçülmedi. Aralık dışındaki veya polifonik olaylarda ölçüm sınırlıdır.

| Kontrol | Güvenilir frame oranı | Medyan F0 |
|---|---:|---:|
| tone_440 | 1.0 | 439.22570878368765 |
| noise | 0.0 | None |
| silence | 0.0 | None |

440 Hz sinüste yaklaşık 439.23 Hz bulundu; gürültü ve sessizlikte bu testin maskesi hiçbir frame'i kabul etmedi. Bu davranış tüm çevresel seslere genellenemez.

Kaynak: [librosa pYIN API](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html). pYIN, Sketch2Sound'un CREPE kontrol çıkarıcısının reprodüksiyonu değildir.

## Çıktılar ve doğrulama

[Frame CSV](../../results/tables/05_pitch_periodicity/frame_features.csv), [run metadata](run.json) ve grafikler `results/figures/05_pitch_periodicity/` altında. Tanımsız özellikler CSV'de boş kaydedilir. Tekrar çalıştırma aynı çıktı dosyalarını günceller.

Kod hücreleri ve notebook şemaları doğrulandı. Koşu Linux/Python 3.12.14 ortamındadır; paket sürümleri run.json'da. Windows/Python 3.11 koşusu henüz doğrulanmadı. Kalite, model controllability veya disentanglement değerlendirmesi yapılmadı.
