# sound-gen — parametrik, çok kategorili Foley üretimi

**Kategori + parametreler → gerçekçi ses.** Ayak sesi, silah, patlama, yağmur, rüzgâr, kuş, araba, at dörtnalı…
Tek bir base model bütün kategorileri öğrenir. Transfer learning ile kategoriye ya da alt kategoriye özel modeller
türetilir (`footsteps` → `footsteps/snow`). Ses; şiddet, ağırlık, mesafe, ortam, süre ve hız gibi parametrelerle
kontrol edilir.

| Notebook | Ne yapar | GPU |
|---|---|---|
| [01_data](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/01_data.ipynb) | veri setlerini indirir, işler, tek paket yapar | gerekmez |
| [02_train_base](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/02_train_base.ipynb) | tüm kategorilerden base model | A100 önerilir |
| [03_finetune](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/03_finetune.ipynb) | transfer learning: kategori / alt kategori | A100 / L4 |
| [04_generate](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/04_generate.ipynb) | parametrik üretim, değerlendirme, diffusion kıyası | herhangi |

> **Durum (2026-10-17):** İlk base model (base_v1, 11,8k adım) ikna edici değil. Tüm çıktılarda 22050/16 Hz
> katlarında sabit tonlar var ve model koşul taslağını henüz takip etmiyor. Denemeler, ölçümler, geri alınan
> değişiklik ve açık hipotezler: **[docs/PROCESS.md](docs/PROCESS.md)**. Literatür ve sonraki mimari önerisi (hazır vokoder + mel üretici): **[docs/RESEARCH.md](docs/RESEARCH.md)**.

---

## Yaklaşım

```
  kategori ──► embedding ─────────────────────────────┐
                                                       ▼
  referans ses ─► 20 bantlık mel "taslak" ─► parametrik düzenleme ─► koşullu WaveFlow ─► ses
  (veri setinden ya da senin)              (şiddet, mesafe, ortam,     ▲  (ayrıntıyı üretir)
                                            süre, hız, ağırlık)        z ~ N(0, T²) + MC dropout
```

- **WaveFlow** (Ping vd., ICML 2020): ham ses üzerinde çalışan, doğrudan maksimum likelihood ile eğitilen kompakt
  bir flow modeli. Tek GPU'da eğitilir ve gerçek zamandan hızlı örnekler.
- **Düşük çözünürlüklü mel taslak** (Andreu & Villanueva, AIIDE 2022): 10–30 bantlık koşul, kaliteyi eğitim
  setine yakın tutarken çeşitlilik bırakır. Biz bu taslağı aynı zamanda **kontrol yüzeyi** olarak kullanıyoruz.
  Parametreler taslağı deterministik olarak düzenler, model de ayrıntıyı üretir. Bu düzenleme adımı öğrenilmiş bir
  model olmadığı için zincire hata eklemez.
- **Kategori embedding'i:** Makalenin "gelecek çalışma" olarak bıraktığı tek ağda çoklu kategori fikrini uygular.
  Alt kategoriler ebeveynlerinin embedding'inden başlar, transfer learning buna dayanır.

### Neden bu yöntem? (kısa kıyas)

| Yaklaşım | Kalite | Sürekli parametre kontrolü | Veri / hesap | Rolü |
|---|---|---|---|---|
| **WaveFlow + mel taslak** (seçilen) | eğitim setine yakın (Andreu & Villanueva) | taslak düzenleme ile doğrudan | 6–25M parametre, saatler | ana yöntem |
| Hazır diffusion, zero-shot (Stable Audio Open) | yüksek, geniş kapsam | yalnız metin; sürekli parametre zayıf | inference ağır | `04`'te kıyas |
| Diffusion fine-tune + kontrol sinyali (Sketch2Sound tarzı) | en yüksek potansiyel | iyi | çok GPU-gün, dikkatli tarif | sonraki adım |
| GAN (WaveGAN / HiFi-GAN) | kısa seslerde iyi | etiket | küçük ama kararsız | — |
| Prosedürel DSP | düşük gerçekçilik | tam | yok | — |

---

## Kategoriler ve veri

| Kategori | Otomatik kaynak | Yaklaşık veri |
|---|---|---|
| footsteps, gunshot, rain, car, dog_bark, keyboard, cough_sneeze | DCASE 2023 T7 (+ ESC-50, FSD50K) | ~600–800 klip/kategori |
| birds, wind, thunder, fire, sea_waves, glass_break, door_knock, helicopter | ESC-50 (40 klip) + **FSD50K** | FSD50K ile yüzlerce |
| explosion | **FSD50K** (+ Freesound) | FSD50K gerekli |
| horse_gallop, jump, footsteps/snow, /gravel, /wood, /concrete | **Freesound API** (ücretsiz anahtar) veya `custom/` | sorgu başına ≤100 |

- Hazır veri setlerinde olmayan sesler için iki yol var: Freesound anahtarıyla otomatik indirme, ya da dosyaları
  `MyDrive/sound-gen/custom/<etiket>/` altına koymak (ör. `custom/footsteps/snow/*.wav`).
- Yeni kategori eklemek için `configs/categories.yaml`'a bir satır yeterli. Yeni parametre eklemek için
  `soundgen/controls.py`'ye bir düzenleme fonksiyonu eklemek yeterli. Model değişmez.

---

## Uygulanan teknikler

| # | Teknik | Nerede | Amaç |
|---|---|---|---|
| 1 | **Transfer learning:** base → kategori / alt kategori, embedding ebeveynden başlatma | `train.py` | az veriyle yeni ses |
| 2 | 3 dondurma stratejisi: `embedding_only` / `last_flows` / `full` (veri miktarına göre otomatik) | `train.py`, `03` | aşırı öğrenmeyi önleme |
| 3 | Ayrımlı öğrenme oranı (embedding ×10) | `train.py` | yeni etiketin hızlı uyumu |
| 4 | **Replay** (ebeveyn kategoriden karışım) + **L2-SP** (base ağırlıklara çekim) | `train.py` | unutmayı (catastrophic forgetting) azaltma |
| 5 | **Dropout** (0.05, gated katmanlar) | `model.py` | küçük veride genelleme |
| 6 | **Monte-Carlo dropout** örneklemede; maske h satır boyunca sabit | `model.py` | ek varyasyon |
| 7 | AdamW + warmup + **cosine** LR | `train.py` | kararlı yakınsama |
| 8 | **EMA** ağırlıklar (0.999); değerlendirme ve üretim EMA ile | `train.py` | daha temiz örnekler |
| 9 | Gradient clipping (1.0) + sonlu olmayan batch'i atlama | `train.py` | flow kararlılığı |
| 10 | **bf16** mixed precision (A100/L4/H100); log-det fp32 | `model.py`, `train.py` | hız / bellek |
| 11 | Weight normalization | `model.py` | WaveFlow makalesindeki gibi |
| 12 | Dequantization gürültüsü (int16 → sürekli) | `train.py` | sessizlikte sonsuz yoğunluğu önleme |
| 13 | **Sınıf dengeli örnekleme** (adet^-0.5) | `data.py` | az verili kategoriler |
| 14 | Olay odaklı kırpma (4 adaydan en yüksek enerjili, %30 rastgele) | `data.py` | sessiz kesitleri azaltma |
| 15 | Augmentation: kazanç (−6…+3 dB), polarite | `data.py` | seviye takibi, genelleme |
| 16 | **Koşul augmentation'ı:** taslak gürültüsü + zamanda bulanıklaştırma | `train.py` | düzenlenmiş taslaklara dayanıklılık |
| 17 | Label dropout (%10 → null etiket) | `train.py` | stil aktarımı, geri dönüş |
| 18 | Sıcaklıkla örnekleme (temperature, varsayılan 0.8) | `model.py` | kalite ↔ çeşitlilik |
| 19 | Post-processing: sabit ton temizleme (WaveFlow artefaktı), high-pass, yumuşak limitleyici | `generate.py` | temiz çıktı |
| 20 | Checkpoint seçimi: en iyi val NLL + 10k/20k/50k kayıtları (erken durdurma) | `train.py` | makalede erken adımlar daha iyi |
| 21 | Değerlendirme: CLAP kategori eşleşmesi, çeşitlilik, kontrol monotonluğu, diffusion kıyası | `evaluate.py`, `04` | ölçülebilir ilerleme |

**Listelenen ama şimdilik uygulanmayanlar:**
- **Öğrenilmiş taslak üretici** (taslak üzerinde küçük bir diffusion): Referans gerektirmeden tamamen yeni taslaklar üretir. En değerli sonraki adım bu.
- **Adversarial fine-tune** (HiFi-GAN ayırt edicileri): Kalite tavanını yükseltir. İkinci aşama.
- **Daha büyük h (32/64) / 44.1 kHz:** Kalite artar, örnekleme yavaşlar. Mevcut veri zaten 22 kHz.
- **SaShiMi/S4 omurga** (Andreu & Villanueva'nın gelecek çalışma önerisi).
- **LoRA / adapter:** Model zaten küçük, kısmi fine-tune daha basit. **SWA:** EMA aynı işi görüyor.
- **Mixup:** Kategoriler karışırsa etiket anlamı bulanıklaşır, bu iş için uygun değil.
- **Hiperparametre araması** (Optuna): Hesap bütçesi gerektirir. Önce temel sonuç alınacak.

---

## Google Colab ile çalışma

### 1. Bir kerelik hazırlık
1. **GitHub token'ı:** github.com → Settings → Developer settings → *Fine-grained tokens* → *Generate new token*
   - Repository access: **Only select repositories** → `sound-gen`
   - Permissions → Repository → **Contents: Read and write**
2. **Colab Secrets:** Colab'de sol paneldeki 🔑 simgesi → *Add new secret*. Her biri için *Notebook access*'i aç.
   - `GITHUB_TOKEN` → yukarıdaki token (çıktıları repoya göndermek için)
   - `FREESOUND_API_KEY` → opsiyonel, https://freesound.org/apiv2/apply
   - `HF_TOKEN` → opsiyonel, Stable Audio Open kıyası için (önce modelin lisansını HF'de kabul et)
3. Drive'da yer: DCASE + ESC-50 paketi ~1 GB. FSD50K ile ~3 GB (FSD50K'nın ham 18 GB'ı yalnız Colab'in
   yerel diskinde durur, Drive'a yazılmaz).

### 2. Notebook'u açmak
Yukarıdaki tablodaki bağlantılar notebook'u doğrudan GitHub'dan açar. Alternatif: colab.research.google.com →
**GitHub** sekmesi → `kaang457/sound-gen` → notebook'u seç.
**Runtime → Change runtime type → A100 GPU** (+ High-RAM). `01` için CPU yeterli.

### 3. Sıra
`01_data` → `02_train_base` → `04_generate`. Alt kategori ya da yeni ses için ek olarak `03_finetune`.
Her notebook'un ilk hücresi (0 · Kurulum) repoyu klonlar ya da günceller, paketi kurar ve Drive'ı bağlar.
Benim repoya yaptığım her güncelleme, bir sonraki çalıştırmada bu hücreyle otomatik olarak Colab'e gelir.

### 4. Oturum düşerse
Eğitim hücresini tekrar çalıştırman yeterli: Drive'daki `runs/<RUN>/last.pt`'den devam eder.

### 5. Çıktıların GitHub'a gelmesi
Son hücreler `colab.push_to_github(...)` ile referans + üretim WAV'larını (≤6 s, 22 kHz/16-bit, dosya başına
~260 KB) ve `metrics.json`'u `samples/<RUN>/...` altına gönderir. Model ağırlıkları ve veri git'e girmez,
Drive'da kalır. Repo **public** olduğu için yalnız paylaşılabilir lisanslı referanslar gönderilir. ESC-50
(CC BY-NC) ve DCASE'in BBC kaynaklı klipleri gönderilmez.

### Drive düzeni
```
MyDrive/sound-gen/
  pack/          audio.bin + index.csv (işlenmiş veri, lisans/atıf dahil)
  custom/        senin seslerin: custom/<etiket>/*.wav
  runs/<RUN>/    best.pt, last.pt, step_*.pt, sketch_bank.pt, log.csv, samples/
```

---

## Repo yapısı
```
soundgen/
  audio.py      mel taslak, ses okuma/yazma
  model.py      koşullu WaveFlow (+ MC dropout, etiket büyütme)
  data.py       indirme (DCASE, ESC-50, FSD50K, Freesound, custom), paketleme, dataset
  train.py      base eğitim + transfer learning, EMA, AMP, checkpoint/resume, sketch bank
  controls.py   parametrik kontroller (taslak düzenleme)
  generate.py   üretim + post-processing + dışa aktarma
  evaluate.py   CLAP metrikleri, kontrol testleri, diffusion kıyası
  colab.py      Drive + GitHub'a gönderme
  smoke.py      sentetik verilerle uçtan uca test (python -m soundgen.smoke)
configs/categories.yaml   kategori taksonomisi ve kaynak eşlemeleri
notebooks/                01-04
samples/                  Colab'den gelen referans + üretim sesleri
```

## Bilinen sınırlar
- Parametreler algısal; fiziksel olarak kalibre değil (ör. "weight" gerçek kilogram değil).
- Kalite, kategorinin verisiyle sınırlı. En güçlü kategoriler DCASE'ten gelenler. ESC-50'de yalnız 40 klibi olan
  kategoriler için FSD50K'yı açmak önemli.
- Üretim varsayılan olarak veri setinden seçilen bir referans taslağa dayanır. Tamamen yeni taslak üretimi
  (öğrenilmiş taslak üretici) sonraki adım.

## Kaynaklar
- Ping, Peng, Zhao, Song. *WaveFlow: A Compact Flow-based Model for Raw Audio.* ICML 2020.
- Andreu, Villanueva Aylagas. *Neural Synthesis of Sound Effects Using Flow-Based Deep Generative Models.* AIIDE 2022.
- Choi vd. *Foley Sound Synthesis at the DCASE 2023 Challenge.* — veri (CC BY 4.0)
- Piczak. *ESC: Dataset for Environmental Sound Classification.* 2015 — veri (CC BY-NC)
- Fonseca vd. *FSD50K.* IEEE/ACM TASLP 2022 — veri (CC)
- Flores García vd. *Sketch2Sound.* ICASSP 2025 — zaman değişkenli kontrol fikri
