# Sketch2Sound — Kontroller ve ilk sonuç okuması

Kaynak: [arXiv:2412.08550v2](https://arxiv.org/html/2412.08550v2), 14 Nisan 2025. İnceleme: 7 Ekim 2026.
Bu not kontrol tanımlarına ve Table I'e odaklanır; tam literatür incelemesi değildir.

## Kontrol sinyalleri — Section II-A

| Kontrol | Makaledeki tanım |
|---|---|
| Loudness | A-weighted magnitude spectrogram üzerinden frame bazında RMS |
| Brightness | Spectral centroid; Hz değeri MIDI benzeri gösterime dönüştürülüp 127'ye bölünür |
| Pitch/periodicity | CREPE-tiny olasılıkları; 0.1 altındaki olasılıklar sıfırlanır |

Kontroller latent zaman eksenine hizalanır. Her kontrolün doğrusal projeksiyonu noisy latent'e eklenir; text conditioning korunur (Section II-B).

## Yayınlanan sonuçlar — Table I

| Model | RMS hatası ↓ (dB) | Centroid hatası ↓ (semitone) | Pitch hatası ↓ (semitone) | Text CLAP ↑ |
|---|---:|---:|---:|---:|
| Text-only | 13.41 | 10.34 | 13.91 | 0.273 |
| Loudness + centroid + pitch | 3.60 | 4.43 | 1.49 | 0.211 |

Kontrollü satırda inference median-filter boyutu 10'dur. Değerler yazarların ölçümleridir; biz yeniden üretmedik. Kontrol hataları azalırken metin uyumu düşüyor; tek bir başarı ölçüsü yeterli değil.

## Değerlendirme — Section III

Kontrol hatası L1 ile, sessiz olmayan frame'lerde ölçülür. Pitch değerlendirmesi iki kayıtta da yeterli periodicity bulunan frame'lerle sınırlıdır. Kalite FAD, metin uyumu CLAP ile değerlendirilir.

## Bizim kodla bağlantısı

[02_feature_extraction.ipynb](../../notebooks/02_feature_extraction.ipynb), unweighted RMS ve Hz centroid hesaplar. A-weighting, MIDI dönüşümü, CREPE ve model uygulanmadı.

[Demo sonuçlarımız](../../experiments/02_feature_extraction/README.md): gain ×0.5 ile RMS −6.0206 dB değişti, centroid aynı kaldı. Bu sayılar yayınlanan kontrol hatalarıyla karşılaştırılamaz.

## Araştırma sorusu

Bizim sonraki değerlendirmemiz yalnızca istenen eğrinin izlenmesini değil, diğer özelliklerin değişimini de ölçmeli. Bu not üzerinden bağımsız kontrolün başarıldığı sonucuna varamayız.
