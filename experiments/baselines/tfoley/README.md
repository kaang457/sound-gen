# T-FOLEY ilk pretrained inference

[Notebook 06](../../../notebooks/06_tfoley_inference.ipynb), [çalıştırıcı](../../../src/baselines/tfoley_inference.py), [makale notu](../../../papers/notes/tfoley_baseline.md), [uygunluk kararı](../../../papers/notes/literature_comparison.md).

## Gerçek koşum

7 Ekim 2026, Linux/Python 3.12.14, PyTorch 2.14.1+cpu. CUDA bu ortamda yok; CPU kullanıldı. Kullanıcının Windows/CUDA koşumu henüz doğrulanmadı.

| Ayar / ölçüm | Değer |
|---|---:|
| Sınıf | Footstep |
| Metin prompt | Yok |
| Seed | 42 |
| Sampling adımı | 100 |
| Guidance | 3 |
| Çıktı | 4 saniye, 22050 Hz mono |
| Parametre sayısı | 74092851 |
| Yalnız sampling süresi, CPU | 270.594 s |
| Çıktı RMS | 0.096747 |
| Çıktı tepe mutlak genlik | 2.190589 |
| Tek örnek smoothed-RMS MAE | 0.008100 |
| Mutlak genlik > 1 örnek oranı | 0.000567 |

Referans: 0.6, 1.8, 3.0 s başlangıçlı sentetik üç pulse; gerçek Footstep kaydı değil. Checkpoint resmi ZIP checksum doğrulaması sonrası yüklenmiştir. Resmî kaynak revizyonu ve checkpoint SHA-256 [run.json](first_inference_run.json) içinde. Modelin RFF embedding device yerleşimi düzeltilmiştir; eğitim ve ağırlıklar değişmez.

Tek örnek MAE ortak RMS smoothing ile ölçülen mühendislik tanısıdır; paper'ın sınıf/örnek ortalamalı E-L1 benchmark'ı değildir. RMS ve zamanlama eğrileri eşleşmeyi görsel incelemek için kaydedildi. Sonlu 88200 örnek doğrulandı. Başarılı inference ses kalitesi, sınıf uygunluğu, disentanglement veya hedefe güvenilir uyum anlamına gelmez.

**Peak 1'i aşıyor.** FLOAT WAV değerleri kırpılmadan saklanır; notebook dinleme sırasında normalize eder. Bu yalnız playback gain değişimidir; CSV ve ölçümler normalize edilmemiş çıktıdan hesaplanır. 16-bit PCM'ye doğrudan dönüştürmek clipping yaratabilir.

## Dosyalar

- [Koşum metadata](first_inference_run.json), [tanılar](first_inference_diagnostics.json)
- [RMS frame CSV](../../../results/tables/tfoley/first_inference_rms.csv)
- [Karşılaştırma grafiği](../../../results/figures/tfoley/first_inference.svg)
- WAV ve referans: yerel `results/audio/tfoley/first_inference*.wav`; Git dışında.

## Yerelde çalıştırma

Temel requirements ve mevcut CUDA PyTorch kurulumundan sonra:

```bash
python -m pip install -r requirements-tfoley.txt
python src/baselines/tfoley_inference.py --prepare --device auto --steps 100 --class-name Footstep --seed 42 --output-name local_inference
```

İlk koşum kaynak repoyu `external/T-FOLEY/`, yaklaşık 549 MB checkpoint'i `models/tfoley/` içine indirir. Git gerekli. Resmî kod/ağırlıklar repoya kopyalanmaz. Ağırlıklar CC-BY-4.0; kaynak atıfları [T-FOLEY](https://github.com/YoonjinXD/T-FOLEY) ve [Zenodo kaydı](https://zenodo.org/records/10826692).

`--target-audio data/raw/reference.wav` kendi referansının RMS'ini kullanır; mono/resample ardından dört saniyeye crop/pad yapılır. `--output-name local_inference` paylaşılan ilk koşum kaydını değiştirmez. CPU ve GPU çıktıları bit düzeyinde aynı olmak zorunda değildir.
