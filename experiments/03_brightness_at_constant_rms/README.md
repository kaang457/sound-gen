# 03 — Brightness at Constant RMS

## Amaç

Yüksek frekans katkısını artırırken RMS'i sabit tutup spectral centroid'in değişimini ölçmek. [Notebook](../../notebooks/03_brightness_at_constant_rms.ipynb) bağımsız çalışır; önceki notebook'ların çalıştırılması gerekmez.

## Ayarlar

İki saniyelik 24 kHz mono sinyal: `sin(2π·200·t) + a·sin(2π·2000·t)`. `a = 0, 0.25, 0.5, 1, 2`. Her sinyal tüm kayıt RMS'i 0.15 olacak şekilde ölçeklenir.

RMS için 1200 örneklik dikdörtgen frame, centroid için aynı uzunlukta Hann-window STFT kullanılır. Hop 240, center False. Frame zamanı pencere orta noktasıdır. Frame'ler her bileşenden tam sayıda periyot içerdiği için frame RMS de sabittir; bu özellik genel transient kayıtlarda garanti değildir.

Centroid ağırlıkları STFT büyüklüğüdür. Bu kontrollü örnekte beklenen değer `(200 + a·2000)/(1+a)` olur. Ölçümler float64 sinyaller üzerinde WAV kodlamasından önce yapılır.

## Ölçülen sonuçlar

| Üst bileşen genlik oranı | Kayıt RMS | Ortalama centroid (Hz) |
|---:|---:|---:|
| 0.00 | 0.15 | 200 |
| 0.25 | 0.15 | 560 |
| 0.50 | 0.15 | 800 |
| 1.00 | 0.15 | 1100 |
| 2.00 | 0.15 | 1400 |

Ölçülen centroid, analitik beklentiyle 1e-6 Hz toleransında eşleşti. Kayıt ve frame RMS 1e-10 toleransında sabit kaldı. Tüm tepe genlikleri 1'in altında kaldı.

Sonuç: Bu sinyal ailesinde RMS sabitken centroid monoton artıyor. Bu, bir descriptor davranışı deneyidir; generative modelin bağımsız kontrol başarısı değildir. Aynı RMS algısal loudness'ı eşitlemez. Perceptual brightness veya pitch ölçümü yapılmadı.

## Çıktılar

- [Centroid ve RMS grafiği](../../results/figures/03_brightness_at_constant_rms/centroid_and_rms.svg)
- [Karşılaştırma tablosu](../../results/tables/03_brightness_at_constant_rms/comparison.csv)
- [Frame ölçümleri](../../results/tables/03_brightness_at_constant_rms/frame_features.csv)
- [Run metadata](run.json)
- WAV örnekleri: `results/audio/03_brightness_at_constant_rms/`; yerel olarak oluşturulur ve Git'e eklenmez.

Sesleri ayrı tepe normalizasyonu olmadan dinlemek için oynatıcıda `normalize=False` kullanılır. WAV'lar PCM_24 olarak yazılır; tablolar yazma öncesi sinyal ölçümleridir.

## Literatür bağlantısı

[Sketch2Sound notumuz](../../papers/notes/sketch2sound_controls.md) centroid'i kullanılan kontrol özelliklerinden biri olarak açıklar. Burada yalnızca centroid'in temel davranışı inceleniyor; model, kontrol normalizasyonu veya yayın sonuçları yeniden uygulanmıyor.

## Doğrulama ve tekrar çalıştırma

Notebook şeması ve bütün kod hücreleri doğrulandı. Sabit RMS, centroid artışı ve analitik sonuç eşleşmesi assertion ile kontrol edildi; grafik görsel olarak incelendi.

Bu koşu Linux/Python 3.12.14 ortamında yapıldı; sürümler ve sinyal hash'leri `run.json` içinde. Windows/Python 3.11 koşusu henüz doğrulanmadı.

Aynı dosyalar yeniden çalıştırmada güncellenir. Ayar değiştirilecekse önce mevcut sonuçlar ayrı bir deney klasöründe saklanmalıdır. Bu tablo mevcut ayarlara aittir.
