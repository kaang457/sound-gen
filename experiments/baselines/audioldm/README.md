# AudioLDM: ilk referans-ses koşullu koşum

[Notebook 08](../../../notebooks/08_audioldm_reference_audio.ipynb), [çalıştırıcı](../../../src/baselines/audioldm_reference_inference.py), [kod/makale notu](../../../papers/notes/audioldm_reference_baseline.md).

## Kaynak ve gerçek koşum

InspectorJ, “Bird Whistling, Robin, Single, 13.wav”, [Freesound 456440](https://freesound.org/people/InspectorJ/sounds/456440/), CC-BY-4.0. Önceki 04–05 ile aynı checksum-pinned librosa mono 22050 Hz OGG türevi; orijinal WAV değil. [Kaynak indirme/atıf](../../../src/audio/real_sample.py).

Resmî `audioldm-m-full` checkpoint, pozitif CLAP audio embedding, kullanıcı text prompt yok. Boş metin unconditional branch içinde kalır. Batch/candidate 1, text candidate scoring kapalı, pozitif koşul dropout'u 0. Checkpoint 4.57 GB; MD5 `46bad9f176651404b3cf1484942749b9`. Kaynak revizyonu `4054cb418ba3d947d94f6ad302f1d6a320e2313c`. [Zenodo checkpoint](https://zenodo.org/records/7813012) kaydında CC-BY-4.0; kod [resmî repository](https://github.com/haoheliu/AudioLDM).

| Ayar / ölçüm | Gerçek CPU koşumu |
|---|---:|
| Torch | 2.11.0+cpu |
| Python / sistem | 3.12.14 / Linux |
| Seed / DDIM adımı | 42 / 50 |
| Guidance | 2.5 |
| Kaynak süre | 2.698639 s |
| Koşullama waveform süresi | 5.12 s |
| Son çıktı | 5 s, 16000 Hz mono |
| Üretim hesaplama süresi | 68.747 s |
| Çıktı RMS | 0.001704 |
| Çıktı peak | 0.035359 |
| -50 dBFS RMS eşiğini geçen çıktı frame | 24 / 313 |

Süre CLAP/latent encoding, DDIM ve decoder/vocoder içerir; model yükleme, checksum ve I/O hariç. Vocoder float32 döndürdü; PCM16 dönüşümü bu koşumda uygulanmadı. 81952 decoded sample 80000 sample'a kırpıldı. FLOAT WAV'a peak normalizasyonu uygulanmadı.

## Yorum ve sınır

Üretim tamamlandı, sonlu 80000 sample doğrulandı. Çıktı düşük seviyeli; aynı FFT ölçeğinde belirgin enerji çoğunlukla yaklaşık 2.3–3.6 s arasında. Referansın olay zamanlamasını veya waveform'unu kopyalamıyor. Seçili RMS eşiği audibility/algısal kalite sınırı değildir.

Öznitelik tanıları ses karakteri/kategori doğruluğu, FAD, dinleme skoru veya bağımsız kontrol başarısı göstermez. Burada dinleme/kategori başarı değerlendirmesi yapılmadı. Tek seed/tek örnek; kuş türü doğruluğu ve eğitim verisiyle örtüşme denetlenmedi. Henüz ana model seçimini kesinleştirmiyoruz.

Notebook ortak gain ile referans/çıktıyı dinletir; ayrıca düşük seviyeli çıktıyı içerik incelemesi için ayrı normalize eder. Bu son player level karşılaştırması için kullanılmaz. Model doğrudan RMS/pitch/brightness slider'ları sağlamaz; T-FOLEY RMS error'u ile model sıralaması yapılmaz.

## Dosyalar

- [Gerçek koşum metadata](robin_reference_first_run.json)
- [Sinyal tanıları](../../../results/tables/audioldm/robin_reference_first_summary.json)
- [Referans frame CSV](../../../results/tables/audioldm/robin_reference_first_reference_frames.csv), [çıktı frame CSV](../../../results/tables/audioldm/robin_reference_first_generated_frames.csv)
- [Spektrogram](../../../results/figures/audioldm/robin_reference_first.svg)
- Yerel WAV: `results/audio/audioldm/robin_reference_first*.wav`; Git dışında.

## Windows kurulum ve çalıştırma

Proje root'unda ayrı ortam:

```powershell
python -m venv .venv-audioldm
.\.venv-audioldm\Scripts\python.exe -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-audioldm\Scripts\python.exe -m pip install -r requirements-audioldm.txt
```

Notebook mevcut `.venv` kernel'inde kalabilir; model subprocess yeni ortamdan çalışır. İlk `--prepare` resmî kaynak ve 4.57 GB checkpoint indirir; yarım checkpoint indirmesi resume eder. Kaynak revizyon/checksum doğrulanır. Tokenizer/config küçük dosyalardır; bütün RoBERTa ağırlıkları zaten AudioLDM checkpoint'inden yüklenir, ek encoder checkpoint indirilmez. Linux CPU koşumu test edildi; Windows/CUDA inference henüz doğrulanmadı. Tekrar koşum `local_robin_reference` dosyalarına yazar ve ilk CPU kaydını korur.
