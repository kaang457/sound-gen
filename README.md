# sound-gen — istekten oyun sesi

**Bir istek yaz, model sıfırdan ses üretsin; 8 aday arasından seç, DAW düğmeleriyle düzenle, indir.**
İleride aynı model oyun videosunu izleyip sahneye uygun sesleri (ayak, at, yağmur, silah…) kaynak kaynak üretecek.
Hedef: Unreal için bir ses aracı.

| Notebook | Ne yapar | GPU |
|---|---|---|
| [01_data](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/01_data.ipynb) | veriyi indirir, 48 kHz paket yapar (`pack48_poc`) | gerekmez |
| [02_train](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/02_train.ipynb) | özellikleri çıkarır, modeli eğitir, sonra test sayfasını GitHub'a yollar | A100 önerilir |
| [03_studio](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/03_studio.ipynb) | tarayıcıda stüdyo: istek → 8 aday → düzenle → mono / stereo WAV | herhangi bir GPU |
| [04_stable_audio](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/04_stable_audio.ipynb) | **hazır model, eğitim yok:** Stable Audio 3 Small-SFX ile ~30 oyun sesi referans seti (GitHub'a sayfa + tablo) ve aynı stüdyo | herhangi bir GPU |
| [05_video_to_audio](https://colab.research.google.com/github/kaang457/sound-gen/blob/main/notebooks/05_video_to_audio.ipynb) | **sessiz gameplay videosu → sesli video:** Qwen3-VL sahneyi izler, iz planı çıkarır; Stable Audio 3 her izi üretir; arayüzde her iz ayrı: ses, sustur, solo, konum, zaman, istek → yeniden üret | L4 / A100 |

Kendi modelimiz için sıra: **01 → 02 → 03**. Hemen sonuç için: **04** ve **05** (tek başlarına çalışır).

## Videodan ses (hazır modellerle)

```
sessiz video ─► Qwen3-VL (Apache-2.0): "hangi sesler, nerede, ne zaman" → iz planı (JSON, düzenlenebilir)
             ─► hareket enerjisi: olay zamanları görüntüdeki vuruşa / adıma hizalanır
plan ────────► Stable Audio 3 Small-SFX: iz başına ses (ortam döngülü, sürekli, olaylar her biri farklı)
             ─► mikser (seviye, uzaklık, stereo konum) → izler + stereo miks → ffmpeg → sesli mp4
```
Eğitim yok, harici API yok; ağırlıklar indikten sonra internetsiz çalışır. Kod: `soundgen/v2a.py`.
Sınır: olaylar zamanında, ama bir ayak sesi klibinin içindeki adımlar her adıma kare kare oturmaz (ileride senkron modeli).

## Hızlı yol: hazır açık ağırlıklı model (Stable Audio 3)

Elimizdeki veri sıfırdan eğitime yetmediği için ilk ürün yolu hazır bir model: **Stable Audio 3 Small-SFX**
(Stability AI, Mayıs 2026). ~1,28 M lisanslı / CC kayıtla eğitilmiş, ağırlıkları açık. Bir kez indirilir, sonra
internetsiz, kapalı ortamda çalışır; harici API gerekmez.

- Lisans: Stability AI Community License. Yıllık geliri 1 M $ altı kuruluşa ticari kullanım dahil ücretsiz,
  çıktılar kullananın; üstü kurumsal lisans. Metin kodlayıcı Gemma şartlarıyla gelir. Ürüne girmeden lisans metni okunmalı.
- Sonraki adım: ticari kullanılabilir veriyle (CC0 / CC-BY + şirketin kendi kayıtları) **LoRA ince ayarı**. Temel
  ağırlıklar donuk kalır, 50–200 MB'lık ek dosyayı biz eğitiriz.
- Kod: `soundgen/sa3.py`; stüdyo aynı (`Studio(generator=sa3.SA3())`).

## Mimari

```
 istek ("gunfight in the rain")  [+ ileride oyun videosu]
   │  CLIP metin kodlayıcı (hazır, donuk) → 77 token'lık vektör dizisi
   ▼
 ÜRETİCİ (bizim eğittiğimiz): flow matching çok-koşullu transformer (MMAudio mimarisi)
   │  3 başlangıç gürültüsü + aralarında 5 ara değer → 8 aday, 25 adımda
   ▼
 mel-VAE çözücü + BigVGAN-v2 (hazır) → 44,1 kHz ses
   ▼
 efekt rafı (bizim): şiddet · uzaklık · ortam (reverb) · stereo konum · süre → WAV
```

- Üretici [MMAudio](https://github.com/hkchengrex/MMAudio)'nun ağı (Sony, CVPR 2025; kod MIT). Kodunu kütüphane olarak
  kullanıyoruz, **ağırlıkları kendi verimizle sıfırdan eğitiyoruz**.
- Video koşulu modelde şimdiden var ama boş; faz 2'de aynı model video ile eğitilecek (ikinci bir sistem yok).
- Karışımlar iki sesin üst üste binmesi değil: iki varyasyonun başlangıç gürültüsünün arasından üretilen yeni bir ses.

Ayrıntı, alternatifler, yol haritası, veri ve bütçe: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
Buraya nasıl gelindiği (WaveFlow, taslak modeli, CLAP denemeleri): **[docs/RESEARCH.md](docs/RESEARCH.md)** ve
[`research-v1`](https://github.com/kaang457/sound-gen/tree/research-v1) dalı.

## Durum (24 Ekim 2026)

| | |
|---|---|
| Kod | özellik çıkarma, eğitim, 8 adaylı üretim, test sayfası, stüdyo hazır; CPU'da uçtan uca test geçti (küçük ağ, sentetik veri) |
| İlk eğitim | `mm_v1`: araştırma verisiyle (~48 saat, lisans filtresiz, ticari kullanılamaz) — sonuç `results/mm_v1/` |
| Hazır model | `04_stable_audio`: Stable Audio 3 Small-SFX referans seti — sonuç `results/sa3_ref_v1/` |
| Videodan ses | `05_video_to_audio`: Qwen3-VL + Stable Audio 3, eğitimsiz |
| Sıradaki | ticari veriyle Stable Audio 3 LoRA ince ayarı; videodan ses için GUI; lisanslı veri teklifleri (Pro Sound Effects, BOOM, Shutterstock); faz 2: video |

## Kullanılan hazır parçalar ve lisansları

| Parça | Lisans | Not |
|---|---|---|
| MMAudio kodu | MIT | ağ, VAE ve flow matching kodu |
| MMAudio 44,1 kHz mel-VAE (`v1-44.pth`) | MMAudio sürümüyle (ticari kullanım için doğrulanmalı) | ticari üründe kendi VAE'mizle değiştirilebilir (etiketsiz veriyle eğitilir) |
| BigVGAN-v2 (NVIDIA) | MIT, "ticari kullanıma hazır" | vocoder |
| Qwen3-VL-8B / 4B-Instruct (Alibaba) | Apache-2.0 | videodan ses planı (05) |
| Stable Audio 3 Small-SFX (Stability AI) | Stability AI Community License (1 M $ gelir altı ticari serbest) | hazır model (04); T5Gemma metin kodlayıcı Gemma şartlarıyla |
| DFN5B CLIP ViT-H/14 (Apple) | model kartına bakılmalı | metin kodlayıcı |
| Eğitim verisi `poc_max` | karışık (NC dahil) | sadece kavram kanıtı |

## Repo yapısı
```
soundgen/
  mm/setup.py      MMAudio kodu (sabit sürüm) + harici ağırlıklar
  mm/features.py   paket → açıklamalar → VAE latentleri + CLIP metin token'ları (önbellek)
  mm/train.py      flow matching eğitimi (GPU'da, veri yükleyicisiz), EMA, devam etme, zaman bütçesi
  mm/generate.py   istek → 3 varyasyon + 5 karışım → ses
  v2a_ui.py        video stüdyosu (Gradio): iz başına düzenleme, zaman çizelgesi
  v2a.py           sessiz video → iz planı (Qwen3-VL) → izler (SA3) → miks → mp4; tek iz yeniden üretme
  sa3.py           hazır model: Stable Audio 3 Small-SFX (kurulum, erişim, üretim, referans seti + sayfa/tablo)
  checks.py        ezber testi, CLAP tanınabilirlik, olay sayısı, test sayfası + zip
  fx.py            efekt rafı (sinyal işleme)
  studio.py, studio_ui.py   stüdyo motoru + Gradio arayüzü
  data.py          veri indirme ve 48 kHz paket
  audio.py, evaluate.py, colab.py, smoke.py (python -m soundgen.smoke [--mm])
notebooks/         01_data, 02_train, 03_studio, 04_stable_audio, 05_video_to_audio
docs/              ARCHITECTURE.md, RESEARCH.md, NOTEBOOKS.md
results/           koşu başına test sayfası, sesler, ölçümler (modeller Drive'da)
```

## Colab
1. Bir kerelik: GitHub'da fine-grained token (yalnız `sound-gen`, *Contents: Read and write*) → Colab 🔑 Secrets →
   `GITHUB_TOKEN`.
2. Notebook'u tablodaki bağlantıdan aç, **Runtime → A100 GPU**. Her notebook'un ilk hücresi repoyu günceller ve Drive'ı bağlar.
3. Kesilirse aynı hücreyi tekrar çalıştır; özellik çıkarma ve eğitim kaldığı yerden devam eder.
