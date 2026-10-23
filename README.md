# sound-gen — sıfırdan oyun sesi üretimi

**İstek → yeni ses, referans kayıt olmadan.** "laser gunshot", "spaceship engine hum", "zombie growl" gibi bir
istek (ya da bir kategori, karışım, elle girilen vektör) verilir; model her seferinde yeni bir ses üretir. Sonra
ses bir DAW eklentisi gibi düzenlenir (şiddet, uzaklık, ortam, stereo konum, süre) ve mono / stereo WAV indirilir.
Hedef: Unreal için bir ses aracı.

| Notebook | Ne yapar | GPU |
|---|---|---|
| [01_data](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/01_data.ipynb) | veriyi indirir, 48 kHz paket yapar (`pack48_poc`) | gerekmez |
| [03_mel_generator](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/03_mel_generator.ipynb) | **bizim modelimizi** eğitir; sonra sıfırdan üretir, test eder, sonuçları GitHub'a yollar | A100 / L4 |
| [04_studio](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/04_studio.ipynb) | tarayıcıda açılan stüdyo (GUI): üret, varyasyonlar + karışımlar, düzenle, indir | herhangi bir GPU |
| [02_vocoder](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/02_vocoder.ipynb) | (opsiyonel) kendi WaveFlow vocoder'ımız, BigVGAN'la kıyas için | A100 |

Sıra: **01 → 03 → 04**. Eski poc yolunun notebook'ları `notebooks/legacy/` altında.

---

## Mimari (rota v3, 23 Ekim)

```
 "spaceship engine hum"  (ya da kategori / karışım / elle vektör / örnek ses)
   │  CLAP (hazır, dondurulmuş): istek → 512 sayılık vektör
   ▼
 [BİZİM] mel üretici: diffusion, gürültüden vektöre uyan YENİ bir mel çizer        soundgen/sketch_model.py
   │      (4 varyasyon = 4 farklı başlangıç gürültüsü; karışım = iki gürültünün küresel ortası)
   ▼
 [BİZİM] üretim kontrolleri: olay sayısı (tek atış → çatışma)                       soundgen/controls.py
   ▼
 BigVGAN-v2 (hazır, NVIDIA, MIT): mel → 44,1 kHz ses                               soundgen/vocoder.py
   ▼
 [BİZİM] efekt rafı: şiddet, uzaklık, ortam (sentetik IR reverb), stereo, süre      soundgen/fx.py
   ▼
  mono / stereo WAV                                                                soundgen/studio*.py
```

- **Bizim eğittiğimiz: mel üretici.** Sesin ne olduğu, zamanlaması, kaç olay olduğu ve spektrumu orada doğar.
  Eğitimde her klip kendi CLAP *ses* vektörüyle eşleşir: etiket gerekmez, paketteki her ses öğrenilir. Kategorisi
  bilinen kliplerde bazen kategori adının *metin* vektörü kullanılır ki yazılı istekler isabetli olsun.
- **Mel tanımı BigVGAN-v2 ile birebir aynı:** 44,1 kHz, n_fft 2048, hop 512, 128 bant, 0–22,05 kHz (fark ~1e-6).
  Kendi WaveFlow'umuz da aynı tanımla eğitilebilir; iki vocoder birbirinin yerine takılır.
- **Efektler üretimden sonra, sesin üzerinde:** kesin ("−6 dB" gerçekten −6 dB) ve anında.

Gerekçeler ve kararların geçmişi: [docs/PROCESS.md §11–§12](docs/PROCESS.md). Notebook'ların hücre hücre açıklaması:
[docs/NOTEBOOKS.md](docs/NOTEBOOKS.md).

## Sıfırdan üretim nasıl test ediliyor (03, eğitimden sonra otomatik)

Test seti: 8 kategori × 2 tohum, 3 karışım, 8 serbest istek (uzay gemisi, lazer silahı, kılıç, canavar…), 2 elle
vektör, üretilmiş bir silah sesine kontroller (çatışma, uzak, mağara). Her ses için:
1. **Ezber mi?** Eğitimdeki en yakın kayda uzaklık (her zaman kaydırmasında); ölçüt, modelin hiç görmediği gerçek
   kayıtların aynı uzaklığı. Birebir kopya 0 dB.
2. **Tanınıyor mu?** Ayrı bir CLAP modeliyle (koşullamada kullanılandan farklı): kategori doğruluğu (gerçek kayıtlarınkiyle
   yan yana) ve istek eşleşmesi (ses kendi isteğine diğer isteklerden daha mı yakın).
3. **Dinleme:** mel resmi + ses içeren tek sayfa ve zip → `results/<RUN>/`.

## Veri

Varsayılan `poc_max`: bulunabilen en büyük veri, lisans filtresi yok (DCASE 2023 T7, ESC-50, FSD50K dev + eval;
kategorilere uymayan klipler "other" etiketiyle). Proje çevrimdışı / araştırma amaçlı; bu veriyle eğitilen modelin
çıktıları ticari kullanılamaz. `commercial` seçeneği yalnız CC0 / CC BY klipleri alır. Repo public olduğu için
paylaşılamaz lisanslı gerçek kayıtlar GitHub'a konmaz.

## Durum

| Ne | Sonuç |
|---|---|
| poc_v1 (22 kHz, 20 bantlık taslak, kategori koşullu WaveFlow; 48 saat veri) | Referans taslaktan gerçekçi ses: seviye ±2 dB, zarf korelasyonu 0,84–1,00, kontroller tutuyor (seviye ±1,6 dB, yankı süresi 1,31 / 1,38 s). Ama sesin yapısı referanstan geliyordu. `results/poc_v1/` |
| Rota v3 (istek → CLAP → mel üretici → BigVGAN → efektler) | Kod, notebook'lar ve stüdyo hazır; CPU'da uçtan uca test geçti (BigVGAN rastgele ağırlıkla, CLAP sahte). **Henüz GPU'da eğitilmedi.** |

## Colab ile çalışma

1. **Bir kerelik:** GitHub'da fine-grained token (yalnız `sound-gen`, *Contents: Read and write*) → Colab'de 🔑
   Secrets → `GITHUB_TOKEN` (notebook erişimi açık). Opsiyonel: `FREESOUND_API_KEY`, `HF_TOKEN`.
2. Tablodaki bağlantılar notebook'u GitHub'dan açar. **Runtime → A100 GPU** (01 için CPU yeterli).
3. Her notebook'un ilk hücresi repoyu günceller ve Drive'ı bağlar. Eğitim kesilirse hücreyi tekrar çalıştır:
   Drive'daki `last.pt`'den devam eder. Her koşu bitince sonuçlar GitHub'a `results/<RUN>/` olarak gelir.
4. Drive'da yer: 48 kHz `pack48_poc` ≈ 16–17 GB (+ CLAP vektörleri ~100 MB).
5. Stüdyo (04) Colab'da çalışır ve herkese açık geçici bir bağlantı verir; hücre açık kaldıkça çalışır.

```
MyDrive/sound-gen/
  pack48_poc/     audio.bin + index.csv + meta.json (48 kHz)
  runs/<RUN>/     checkpoint'ler, log, publish/
```

## Repo yapısı
```
soundgen/
  audio.py         mel sözleşmesi (BigVGAN-v2 ile aynı), ses okuma/yazma
  sketch_model.py  mel üretici (diffusion; CLAP ya da etiket koşullu), CLAP vektörlerinin hesabı
  vocoder.py       BigVGAN-v2 / WaveFlow vocoder, aynı arayüz
  fx.py            efekt rafı: kazanç, uzaklık, sentetik IR reverb, pan, süre, limiter
  studio.py        stüdyo motoru: varyasyonlar, karışımlar, dışa aktarma
  studio_ui.py     Gradio arayüzü
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
notebooks/         01_data, 02_vocoder (opsiyonel), 03_mel_generator, 04_studio; legacy/ (poc yolu)
results/           koşu başına sade sonuç klasörü (eski çıktılar: archive dalı)
docs/              PROCESS.md (karar günlüğü), NOTEBOOKS.md, RESEARCH.md
```

## Sırada
1. GPU'da 01 → 03 ve ilk sıfırdan üretim sayfası; 04 ile stüdyo.
2. Prosedürel sentezle tasarlanmış sesler (sci-fi vb.) için veri üreteci.
3. Kendi WaveFlow vocoder'ımızı (02) BigVGAN'la kıyaslamak; gerçek (kayıtlı) impulse response'lar.

## Kaynaklar
- Ping, Peng, Zhao, Song. *WaveFlow: A Compact Flow-based Model for Raw Audio.* ICML 2020.
- Andreu, Villanueva Aylagas. *Neural Synthesis of Sound Effects Using Flow-Based Deep Generative Models.* AIIDE 2022.
- Choi vd. *Foley Sound Synthesis at the DCASE 2023 Challenge.* arXiv 2304.12521.
- Lee vd. *BigVGAN: A Universal Neural Vocoder with Large-Scale Training.* ICLR 2023 (v2: NVIDIA, MIT).
- Wu vd. *Large-scale Contrastive Language-Audio Pretraining (CLAP).* ICASSP 2023 · Liu vd. *AudioLDM.* ICML 2023.
- Laroche & Dolson. *Improved phase vocoder time-scale modification of audio.* 1999.
- Ho vd. *Denoising Diffusion Probabilistic Models.* 2020 · Song vd. *DDIM.* 2021 · Ho & Salimans. *Classifier-Free Guidance.* 2022.
- Fonseca vd. *FSD50K.* IEEE/ACM TASLP 2022 · Piczak. *ESC-50.* 2015.
