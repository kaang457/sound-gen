# 01 — Audio Basics

## Amaç ve durum

WAV → waveform → spectrogram → mel-spectrogram akışını öğrenmek ve çıktıları tekrar üretilebilir şekilde kaydetmek.
Kod: [01_audio_basics.ipynb](../../notebooks/01_audio_basics.ipynb).
Bu bir temel sinyal işleme deneyidir. Model eğitimi, inference veya makale reprodüksiyonu yapılmadı.

## Kurulum ve çalıştırma

Proje kökünden `python -m pip install -r requirements.txt` ile temel bağımlılıklar kurulabilir.
Windows/VS Code'da mevcut Python 3.11 `.venv` kernel'ini seç ve notebook'ta Run All kullan. Ortamı tekrar oluşturmak gerekmez.
`AUDIO_PATH = None` internetten veri indirmeden demo üretir. Kendi WAV dosyanı seçmek için bu değişkeni değiştir.

Notebook varsayılan olarak aynı çıktı dosyalarını günceller. Yeni kayıtla karşılaştırmadan önce eski çıktıların kopyasını farklı bir deney klasöründe sakla. Aşağıdaki değerler yalnızca kayıtlı demo çalıştırmasına aittir; başka bir girişle README tablosunu da güncelle.

## Giriş ve ayarlar

- Sentetik 3 saniyelik mono darbe dizisi; seed = 42.
- Darbe başlangıçları: 0.4, 1.2, 2.0 s; sönme zaman sabiti: 0.12 s.
- Sinüs bileşenleri: 220 Hz ve 1400 Hz; ayrıca gürültü.
- Demo yazılırken tepe 0.8'e ölçeklendi; WAV PCM_16 olarak kaydedilip tekrar okundu.
- Örnekleme hızı: 24 000 Hz. Model önerisi veya standart dataset hızı değildir.
- FFT/pencere: 1024 örnek, Hann; hop: 256 örnek; center: True.
- 64 mel bandı; 0–12 000 Hz; güç = STFT büyüklüğünün karesi.
- Her temsil kendi maksimumuna göre dB'ye çevrildi, top_db = 80.

## Ölçülen sonuçlar

| Ölçüm | Bu çalıştırma |
|---|---:|
| Süre | 3.000 s |
| Mono örnek sayısı | 72000 |
| Tepe genliği | 0.79998779 |
| RMS | 0.08395620 |
| Pencere süresi | 42.6667 ms |
| Hop süresi | 10.6667 ms |
| FFT bin aralığı | 23.4375 Hz |
| STFT boyutu | 513 × 282 |
| Mel boyutu | 64 × 282 |
| Genlik ×0.5 sonrası RMS oranı | 0.5000 |
| Genlik ×0.5 sonrası seviye farkı | -6.020600 dB |

Genliği yarıya indirince RMS yarıya indi. Bu işlem süreyi ve frekans konumlarını değiştirmedi; algısal loudness veya model disentanglement sonucu ölçülmedi.
Her spectrogram kendi maksimumuna göre ölçeklendiği için grafik rengi kayıtlar arası mutlak seviye karşılaştırması sağlamaz.

## Çıktılar

- [Waveform](../../results/figures/01_audio_basics/waveform.svg)
- [Spectrogram](../../results/figures/01_audio_basics/spectrogram.svg)
- [Mel-spectrogram](../../results/figures/01_audio_basics/mel_spectrogram.svg)
- [Sayısal sonuçlar](../../results/tables/01_audio_basics/summary.csv)
- [Run metadata](run.json): UTC tarih, giriş dosyası SHA-256, analiz ayarları ve paket sürümleri.
- Demo WAV: `data/generated/audio_basics_demo.wav`; Git'e eklenmez, notebook yeniden oluşturur.

## Literatür bağlantısı

[Kod–literatür eşleştirmesi](../../papers/notes/code_literature_map.md) her çalışmayla ilişkiyi ve eksik parçaları açıklar.
Yakın hazırlık alanları: Sketch2Sound ve Audio ControlNet için seviye/zaman analizi; T-FOLEY için waveform ve temporal olayları görme.
Bu yöntemlerin kontrol çıkarıcıları ve modelleri burada uygulanmadı. Yayınlardaki sayısal sonuçlar bu deneyin sonuçları değildir.

## Doğrulama ve sınırlar

Notebook şeması doğrulandı ve tüm kod hücreleri sırasıyla çalıştırıldı. Spektrum boyutları ve sonlu değerler assertion ile kontrol edildi. Grafikler ve CSV/JSON çıktıları üretildi.
Doğrulama ortamı: Linux, Python 3.12.14; ayrıntılı paket sürümleri `run.json` içinde.
Kullanıcının Windows/Python 3.11 ortamında henüz çalıştırıldığı doğrulanmadı.

Sentetik demo gerçek Foley kalitesi hakkında çıkarım sağlamaz. FAD/MOS, pitch/brightness ölçümü, otomatik onset tespiti veya generative model kontrol başarısı değerlendirilmedi.
