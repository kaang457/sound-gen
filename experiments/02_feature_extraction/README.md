# 02 — Feature Extraction

## Amaç

[Notebook](../../notebooks/02_feature_extraction.ipynb), kayıt boyunca frame bazında RMS ve spectral centroid çıkarır. Bu ortak analiz altyapısıdır; Sketch2Sound modeli veya kontrol çıkarıcılarının tam reprodüksiyonu değildir.

## Çalıştırma

VS Code'da mevcut Python 3.11 `.venv` kernel'i seçilip notebook hücreleri sırayla çalıştırılır. `AUDIO_PATH = None` ile demo bağımsız olarak oluşturulur; 01 notebook'un önce çalışması gerekmez. Kendi kaydı için `AUDIO_PATH` bir WAV yolu olarak ayarlanır. Yeni paket kurulumu gerekmez.

Aynı çıktı yolları sonraki çalıştırmada güncellenir. Farklı kayıtlarla karşılaştırmadan önce önceki çıktıları ayrı bir deney klasöründe sakla. Bu notun sonuçları aşağıdaki sentetik demo çalıştırmasına aittir.

## Giriş ve ayarlar

- 3 saniye, 24 kHz, mono, PCM_16; RNG seed 42.
- Darbe başlangıçları: 0.4, 1.2, 2.0 s; sönme sabiti 0.12 s.
- Her darbe: 220 ve 1400 Hz sinüs bileşenleri + geniş bant gürültü.
- Frame/FFT uzunluğu: 1024; hop: 256; center: True; sıfır padding.
- RMS: waveform üzerinde dikdörtgen frame; dB referansı dijital genlik 1.
- Centroid: Hann-window STFT büyüklüğüyle ağırlıklandırılmış frekans ortalaması.
- Centroid grafiği/özeti için RMS > −60 dBFS maskesi.
- Genlik ×0.5 karşılaştırmasında özgün sinyalin frame maskesi korunur.

## Ölçülen sonuçlar

| Ölçüm | Bu çalıştırma |
|---|---:|
| Frame sayısı | 282 |
| Eşik üstü frame sayısı | 204 |
| Frame süresi | 42.6667 ms |
| Hop süresi | 10.6667 ms |
| Frame RMS ortalaması, tüm frame'ler | 0.04349962 |
| Maksimum frame RMS | 0.28111978 |
| Ortalama centroid, eşik üstü frame'ler | 4637.4956 Hz |
| Genlik ×0.5 sonrası ortalama RMS seviye farkı | −6.020600 dB |
| Aynı frame'lerde maksimum centroid farkı | 0.000000 Hz |

Frame RMS değerlerinin aritmetik ortalaması, tüm kaydın RMS'iyle aynı ölçüm değildir. Gürültü bileşeni yüksek frekans kutularına yayıldığından centroid sinüs frekanslarından daha yüksek olabilir; centroid pitch tahmini değildir.

Genlik deneyi feature davranışını kontrol eder. Bu sonuç bir generative modelin controllability veya disentanglement performansını göstermez.

## Çıktılar

- [Özellik eğrileri](../../results/figures/02_feature_extraction/feature_curves.svg)
- [Frame değerleri](../../results/tables/02_feature_extraction/frame_features.csv)
- [Özet ölçümler](../../results/tables/02_feature_extraction/summary.csv)
- [Run metadata](run.json): zaman, giriş hash'i, ortam ve analiz ayarları.

Tanımsız/maskelenmiş centroid değerleri frame CSV'sinde boş bırakılır. Grafiklerde bu frame'ler çizilmez. Eşik bir event/onset dedektörü değildir.

## Literatür bağlantısı

[Sketch2Sound, Section II-A](https://arxiv.org/html/2412.08550v2#S2.SS1) loudness, spectral centroid ve pitch/periodicity kontrol sinyallerini kullanır. Bu notebook yalnızca ilgili temel özellikleri inceler: waveform RMS A-weighted loudness ile aynı değildir; centroid Hz olarak kalır. CREPE, centroid normalizasyonu, model veya eğitim uygulanmadı.

[Kod–literatür eşleştirmesi](../../papers/notes/code_literature_map.md) uygulanan ve eksik bileşenleri ayırır. Yayın sonuçları bu demo sonuçlarından ayrıdır.

API tanımları: [librosa resmi feature kaynağı](https://librosa.org/doc/0.11.0/_modules/librosa/feature/spectral.html).

## Doğrulama ve sınırlar

Tüm notebook hücreleri çalıştırıldı ve notebook şeması doğrulandı. RMS/STFT frame sayıları eşleşti. Genlik ×0.5 testi RMS oranını ve centroid invariance'ını doğruladı. Bilinen 1500 Hz sinüsle RMS ve centroid hesapları; sıfır sinyalle sessizlik maskesi kontrol edildi. Üç panelli grafik görsel olarak incelendi.

Bu koşu Linux/Python 3.12.14 ortamında yapıldı; tam sürümler `run.json` içinde. Kullanıcının Windows/Python 3.11 ortamında bu notebook henüz doğrulanmadı. 01 notebook'un yerelde çalıştığı kullanıcı tarafından doğrulandı.

−60 dBFS eşiği bu deneyin seçimidir; makalenin evaluation protokolü değildir. Sessizlik yakınında centroid güvenilirliği azalır. Pitch, perceptual loudness, kalite metriği veya model kıyaslaması yoktur.
