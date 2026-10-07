# Gerçek Footstep referansı ile ikinci koşum

[Notebook 07](../../../../notebooks/07_tfoley_real_footstep.ipynb), [referans hazırlama](../../../../src/audio/footstep_sample.py), [tanı kodu](../../../../src/audio/tfoley_comparison.py).

## Kaynak ve ayarlar

freefire66, “footsteps.wav”, [Freesound 175954](https://freesound.org/people/freefire66/sounds/175954/), CC0-1.0. Kaldırımda bot adımları. Kamuya açık HQ MP3 önizlemesi; orijinal WAV indirilmedi. Checksum doğrulandı. Sabit 5–9 saniye bölümü mono 22050 Hz FLOAT WAV'a dönüştürüldü; peak normalizasyonu yapılmadı. Kaynak atıf/checksum/dönüşümler [comparison.json](real_footstep_inference_comparison.json) içinde.

Model: resmî T-FOLEY EMA checkpoint, Footstep class, seed 42, guidance 3, 100 adım. 2026-10-07 Linux/Python 3.12.14, PyTorch 2.14.1+cpu, CPU. Kaynak revizyon ve checkpoint hash'i [run.json](../real_footstep_inference_run.json) içinde.

## Ölçülen tek çift

| Ölçüm | Değer |
|---|---:|
| Üretim süresi, CPU | 152.045 s |
| Süre | 4 s |
| Smoothed RMS MAE | 0.005811 |
| Smoothed RMS Pearson | 0.762540 |
| Referans global RMS | 0.023300 |
| Çıktı global RMS | 0.020229 |
| Çıktı peak | 0.216502 |
| Seçili çıkarıcıyla referans RMS tepe sayısı | 8 |
| Seçili çıkarıcıyla çıktı RMS tepe sayısı | 10 |

Frame 512/hop 128, resmî koşullamaya uygun elliptic smoothing. RMS peak için relative prominence 15% ve minimum distance 250 ms. Bu çıkarıcının bulduğu tepeler gerçek ayak adımı sayısı veya onset ground truth'u değildir. Referansın 0.702 s tepesine en yakın çıktı tepesi 0.848 s; yaklaşık +145 ms fark var. En yakın eşleme aynı çıktı tepesini tekrar kullanabilir.

Bizim yorumumuz: enerji zarfında benzerlik var, kusursuz temporal kopya yok. Korelasyonun yüksek olması timbre/semantik veya dinlemede benzerliği kanıtlamaz. Bu koşum için algısal dinleme skoru, kategori doğruluğu, FAD ve çoklu-seed benchmark yapılmadı. İlk sentetik koşumdaki [kullanıcı geri bildirimi](../listening_feedback.md) bu yeni koşuma aktarılmaz.

## Çıktılar ve yerel kullanım

- [Frame CSV](../../../../results/tables/tfoley_real_footstep/real_footstep_inference_rms.csv)
- [Grafik](../../../../results/figures/tfoley_real_footstep/real_footstep_inference.svg)
- [Tanılar / provenance](real_footstep_inference_comparison.json), [model koşumu](../real_footstep_inference_run.json)

WAV dosyaları Git dışında. Notebook 07 ilk hücrelerde referansı indirip dinletir; üretim hücresi `local_real_footstep.wav` oluşturur. Sonraki iki player referans/çıktıya aynı gain uygular; kayıtlar değiştirilmez. CSV/metadata/grafik `local_real_footstep` adıyla ilk CPU koşumunu koruyarak yazılır. Temel requirements + mevcut CUDA PyTorch + `requirements-tfoley.txt` kullanılır. Windows/GPU koşumu henüz doğrulanmadı.

Bu deney T-FOLEY'nin kendi desteklediği sınıfta temporal koşullamayı inceler. Kuş/çevresel referansın ses karakteri için sıradaki aday [AudioLDM audio-to-audio](../../../../papers/notes/reference_audio_selection.md); henüz çalıştırılmadı.
