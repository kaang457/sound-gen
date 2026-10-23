# sound-gen — sıfırdan oyun sesi üretimi

**Vektör → yeni ses, referans kayıt olmadan.** Bir kategori vektörü (ör. `gunshot`), birkaçının karışımı
(`gunshot` + `thunder`) ya da elle girilen bir vektör verilir; model her seferinde yeni bir ses üretir. Ses
şiddet, uzaklık, ortam, süre, hız ve olay sayısı (ör. tek atış → çatışma) gibi parametrelerle kontrol edilir.
Hedef: Unreal için bir ses aracı. İleride vektörü bir metin kodlayıcı (CLAP / T5) verecek.

| Notebook | Ne yapar | GPU |
|---|---|---|
| [01_data](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/01_data.ipynb) | veriyi indirir, 48 kHz paket yapar (`pack48_poc`) | gerekmez |
| [02_vocoder](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/02_vocoder.ipynb) | WaveFlow'u evrensel vocoder olarak eğitir (mel → ses) | A100 önerilir |
| [03_mel_generator](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/03_mel_generator.ipynb) | mel üreticiyi eğitir; sıfırdan üretir ve test eder (ezber mi, tanınıyor mu, dinleme sayfası) | A100 / L4 |

Sıra: **01 → 02 → 03**. Eski poc yolunun notebook'ları (referans taslak + kategori modeli, transfer learning)
`notebooks/legacy/` altında; rota v2'de kullanılmıyor.

---

## Mimari (rota v2, 21 Ekim)

```
 vektör (kategori / karışım / elle; ileride metin kodlayıcı)
   │
 [1] mel üretici: diffusion modeli, gürültüden yeni bir mel spektrogram üretir     soundgen/sketch_model.py
   │      sesin NE olduğu, zamanlaması, olay sayısı, kaba spektrumu burada doğar
 [2] kontroller: şiddet, uzaklık, ortam, süre, hız, olay sayısı (kural tabanlı)     soundgen/controls.py
   │
 [3] vocoder: WaveFlow, yalnız mel'i görür (kategori almaz), dalga formunu üretir   soundgen/model.py
   │
  ses (48 kHz)                                                                      soundgen/scratch.py
```

- **Tek ortak model, tür başına alt model yok.** Bütün kategoriler aynı mel üreticide; kategori yalnız bir vektör.
- **Mel sözleşmesi** (bütün modüller bunu konuşur): 48 kHz, n_fft 2048, hop 480 (100 kare/s), 128 mel bandı,
  20 Hz–24 kHz. Değişirse bütün modüller yeniden eğitilir. `soundgen/audio.py`
- **Vocoder evrensel:** etiket gerekmediği için bütün temiz ses eğitimde kullanılır; eğitim kategorileri dışındaki
  mel'leri (ör. sentetik sci-fi) de çalabilir. Modüller ayrı eğitildiği için vocoder ileride daha hızlısıyla
  (HiFi-GAN, BigVGAN, Vocos) değiştirilebilir.
- **Yaratıcılık birinci aşamada.** DCASE 2023 Task 7'nin en iyi sistemleri, AudioLDM ve Stable Audio da aynı ayrımı
  kullanıyor: mel'i diffusion üretir, vocoder çalar.

Kararların gerekçeleri ve ölçümler: [docs/PROCESS.md §11](docs/PROCESS.md). Notebook'ların hücre hücre
açıklaması: [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md).

## Sıfırdan üretim nasıl test ediliyor (03, hücre 4)

1. **Ezber mi?** Üretilen her mel, eğitim kayıtlarıyla her zaman kaydırmasında kıyaslanır; en yakın kaydın uzaklığı
   (dB) yazılır. Ölçüt: modelin hiç görmediği gerçek kayıtların en yakın eğitim kaydına uzaklığı. Birebir kopya 0 dB.
2. **Tanınıyor mu?** CLAP her sesi kategorilerden birine atar; doğruluk, gerçek kayıtlardaki doğrulukla yan yana.
3. **Dinleme:** her ses için mel resmi + oynatıcı içeren tek sayfa ve bütün seslerin zip'i → `results/<RUN>/`.

## Veri

Varsayılan `poc_max`: bulunabilen en büyük veri, lisans filtresi yok (DCASE 2023 T7, ESC-50, FSD50K dev + eval;
kategorilere uymayan klipler "other" etiketiyle). Proje çevrimdışı / araştırma amaçlı; bu veriyle eğitilen modelin
çıktıları ticari kullanılamaz. `commercial` seçeneği yalnız CC0 / CC BY klipleri alır. Repo public olduğu için
paylaşılamaz lisanslı gerçek kayıtlar GitHub'a konmaz.

## Durum

| Ne | Sonuç |
|---|---|
| poc_v1 (22 kHz, 20 bantlık taslak, kategori koşullu WaveFlow; 48 saat veri) | Referans taslaktan gerçekçi ses: seviye ±2 dB, zarf korelasyonu 0,84–1,00, kontroller tutuyor (seviye ±1,6 dB, yankı süresi 1,31 / 1,38 s). Ama sesin yapısı referanstan geliyordu. `results/poc_v1/` |
| Rota v2 (vektör → mel üretici → vocoder) | Kod ve notebook'lar hazır, CPU'da uçtan uca test geçti. **Henüz GPU'da eğitilmedi.** |

## Colab ile çalışma

1. **Bir kerelik:** GitHub'da fine-grained token (yalnız `sound-gen`, *Contents: Read and write*) → Colab'de 🔑
   Secrets → `GITHUB_TOKEN` (notebook erişimi açık). Opsiyonel: `FREESOUND_API_KEY`, `HF_TOKEN`.
2. Tablodaki bağlantılar notebook'u GitHub'dan açar. **Runtime → A100 GPU** (01 için CPU yeterli).
3. Her notebook'un ilk hücresi repoyu günceller ve Drive'ı bağlar. Eğitim kesilirse hücreyi tekrar çalıştır:
   Drive'daki `last.pt`'den devam eder. Her koşu bitince sonuçlar GitHub'a `results/<RUN>/` olarak gelir.
4. Drive'da yer: 48 kHz `pack48_poc` ≈ 16–17 GB.

```
MyDrive/sound-gen/
  pack48_poc/     audio.bin + index.csv + meta.json (48 kHz)
  runs/<RUN>/     checkpoint'ler, log, publish/
```

## Repo yapısı
```
soundgen/
  audio.py         mel sözleşmesi, ses okuma/yazma
  sketch_model.py  mel üretici (diffusion, vektör girdisi)
  model.py         WaveFlow
  train.py         WaveFlow eğitimi (vocoder=True: evrensel vocoder), EMA, AMP, devam etme
  scratch.py       sıfırdan üretim zinciri, ezber testi, CLAP testi, sayfa + zip
  controls.py      parametrik kontroller (mel düzenleme)
  generate.py      üretim (mel ya da referanstan) + temizleme
  data.py          indirme ve paketleme
  diagnostics.py   eğitim sağlık raporu
  publish.py       koşu sonucunu sade klasöre çevirme
  colab.py         Drive + GitHub
  smoke.py         sentetik verilerle uçtan uca test (python -m soundgen.smoke)
notebooks/         01_data, 02_vocoder, 03_mel_generator; legacy/ (poc yolu)
results/           koşu başına sade sonuç klasörü (eski çıktılar: archive dalı)
docs/              PROCESS.md (karar günlüğü), NOTEBOOKS.md, RESEARCH.md
```

## Sırada
1. GPU'da 01 → 02 → 03 ve ilk sıfırdan üretim sayfası.
2. Prosedürel sentezle tasarlanmış sesler (sci-fi vb.) için veri üreteci.
3. Dondurulmuş metin kodlayıcı (CLAP): vektör yerine kelimeler.

## Kaynaklar
- Ping, Peng, Zhao, Song. *WaveFlow: A Compact Flow-based Model for Raw Audio.* ICML 2020.
- Andreu, Villanueva Aylagas. *Neural Synthesis of Sound Effects Using Flow-Based Deep Generative Models.* AIIDE 2022.
- Choi vd. *Foley Sound Synthesis at the DCASE 2023 Challenge.* arXiv 2304.12521.
- Ho vd. *Denoising Diffusion Probabilistic Models.* 2020 · Song vd. *DDIM.* 2021 · Ho & Salimans. *Classifier-Free Guidance.* 2022.
- Fonseca vd. *FSD50K.* IEEE/ACM TASLP 2022 · Piczak. *ESC-50.* 2015.
