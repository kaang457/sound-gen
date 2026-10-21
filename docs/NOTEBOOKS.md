# Notebook'lar: ne yapıyorlar?

Dört notebook bir zincir oluşturuyor. Ağır kod `soundgen/` paketinde; notebook'lar onu çağırıp sonuçları gösteriyor.

```
01_data ──► pack (Drive) ──► 02_train_base ──► runs/base_v1 (Drive) ──┬──► 03_finetune ──► runs/ft_* (Drive)
                                                                       └──► 04_generate ◄──┘
                                       hepsi ──► results/<RUN>/ (GitHub; vitrin, rapor, model)
```

Her notebook'un **hücre 0'ı aynı**: Repodaki son kodu çeker, paketi Colab'in hazır torch/numpy kurulumuna dokunmadan
kurar ve Drive'ı bağlar. Bu yüzden repoya yapılan her düzeltme, hücre 0 tekrar çalıştırılınca gelir.

---

## 01_data: veriyi indir, işle, paketle

| | |
|---|---|
| Girdi | internetten veri setleri; isteğe bağlı `MyDrive/sound-gen/custom/<etiket>/` |
| Çıktı | `MyDrive/sound-gen/<PACK_NAME>/` (varsayılan `pack_clean`) → `audio.bin`, `index.csv` (klip, etiket, bölüm, lisans, atıf), `meta.json`, `ATTRIBUTION.csv` |
| GPU | gerekmez (CPU yeterli) |
| Süre | FSD50K ≈ 1 saat (çoğu ~20 GB indirme, yerel diske) |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `DATASET`: `poc_max` (varsayılan; lisans filtresiz en büyük veri: DCASE + ESC-50 + FSD50K dev + eval + "other" → `pack_poc`; çıktılar ticari kullanılamaz) · `commercial` (yalnız CC0 / CC BY → `pack_full`) · `manual`. Ayrıca `CATEGORY`, `INCLUDE_OTHER`, `MAX_OTHER`, `PACK_NAME`, `MAX_PER_LABEL` |
| 2 · Sayım | Yalnız FSD50K'nın etiket ve lisans dosyalarını indirir (~10 MB). Kategori başına klip sayısını "CC0 + CC BY" ve "tümü" olarak yazar. `auto` ise en çok ticari-temiz klibi olan kategoriyi seçer. |
| 3 · İndir ve paketle | Ticari modda ESC-50 ve DCASE'i atlar, FSD50K'dan yalnız CC0 / CC BY klipleri alır. Sesleri 22,05 kHz mono'ya çevirir, sessizliği kırpar, ≤ 6 s parçalara böler, tek dosyada paketler. Kayıtlar kaynak dosyaya göre train/val/test'e bölünür. Atıf listesini yazar. |
| 4 · Özet ve dinleme | Kategori × bölüm sayım tablosu, dakika, kategori başına 3 örnek ve atıfları |
| 5 · (Opsiyonel) Referansları gönder | Kategori başına bir gerçek sesi atıf dosyasıyla `results/references/<PACK_NAME>`'e yollar |

**Ticari mod dışı (eski base_v1 paketi):** `COMMERCIAL_ONLY = False`, `USE_DCASE`/`USE_ESC50` açık, `CATEGORY = "all"`,
`PACK_NAME = "pack"`. Bu paket ticari model eğitimi için temiz değil: [RESEARCH.md §7](RESEARCH.md).

---

## 02_train_base: tek ortak model, bitince sade sonuç

| | |
|---|---|
| Girdi | `<PACK>/` (varsayılan `pack_poc`) |
| Çıktı | Drive `runs/<RUN>/` (checkpoint'ler, log, ara örnekler) · GitHub `results/<RUN>/` (`README.md`, `showcase.html`, `health.html`, `model.pt`, `sketch_bank.pt`) |
| GPU | A100 önerilir (L4 / T4 da çalışır, preset otomatik seçilir) |
| Süre | `MAX_HOURS` (varsayılan 5 saat) + yayın ~10 dk |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `RUN` (varsayılan `poc_v1`; aynı adla tekrar çalıştırma devam ettirir), `PACK`, `UPSAMPLER`, `STEPS` (0 = süreden hesapla), `MAX_HOURS`, `FIXES` (taslak normalizasyonu, ActNorm, dropout 0, grad_clip 25), `PRESET` |
| 2 · Kuru deneme | Büyük eğitimden önce bir kez: Drive boş alanı, gerçek veriyle ~10 dk eğitim (kayıp düşüyor mu, NaN var mı), sonra yayın ve GitHub gönderimi (`results/<RUN>_dry/`). Hepsi ✅ ise hücre 3 |
| 3 · Eğit ve yayınla | Paketi yerel diske kopyalar, modeli **sıfırdan** eğitir (1.000 adımda bir doğrulama + checkpoint, 5.000 adımda bir ara örnek, 250 adımda bir sağlık ölçümü). Süre dolunca durur, sonra `colab.publish`: vitrin (v1 son işlemiyle), sağlık raporu, küçültülmüş model ve README'yi `results/<RUN>/`'a gönderir |
| 4 · Kayıp eğrisi | Train ve val NLL grafiği |
| 5 · Sonuçları aç | README'yi ve vitrini notebook'ta gösterir |
| 6 · Başka bir koşuyu yayınla | Örneğin clean_v1 (araba) koşusunu aynı sade biçimde gönderir |

Ayrıntılı açıklama ve kavramlar: sohbet geçmişi ve [PROCESS.md §4](PROCESS.md). **base_v1 sonucu:** 11,8k adım, çıktılar
ikna edici değil (1378 Hz katlarında vızıltı, seviye takibi yok); bkz. PROCESS.md §5.

---

## 03_finetune: transfer learning ile kategori / alt kategori modeli

| | |
|---|---|
| Girdi | `runs/<BASE_RUN>/<BASE_CKPT>` + `pack/` (+ gerekirse `custom/` veya Freesound) |
| Çıktı | `runs/ft_<hedef>/` (aynı yapı) |
| GPU | A100 / L4 |
| Süre | `STEPS` (varsayılan 5.000) |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `BASE_RUN`, `PACK` (base'in paketi; replay buradan), `BASE_CKPT`, `TARGETS` (ör. `footsteps/snow` ya da pakette olan `footsteps`), `STRATEGY`, `STEPS`, `REPLAY` |
| 2 · Hedef veri | Hedef pakette yoksa `custom/<hedef>/`'den (ya da anahtar varsa Freesound'dan) ayrı bir paket kurar. Veri yoksa nereye koyulacağını söyleyip durur. Stratejiyi klip sayısına göre seçer: < 30 → `embedding_only`, < 300 → `last_flows`, daha fazlası → `full`. |
| 3 · Fine-tune | Base ağırlıkları yükler; yeni etiketin vektörünü **ebeveyninden** başlatır (`footsteps/snow` ← `footsteps`). Stratejiye göre katmanları dondurur, unutmayı azaltmak için ebeveyn kategoriden örnek karıştırır (replay). `full`'da düşük LR ve L2-SP kullanır. |
| 4 · Karşılaştır | Aynı taslak ve tohumla base ve fine-tune çıktılarını yan yana çalar |
| 5 · GitHub'a gönder | Karşılaştırmayı `results/ft_*/compare`'e yollar |

Henüz çalıştırılmadı.

---

## 04_generate: parametrik üretim, değerlendirme, kıyas

| | |
|---|---|
| Girdi | `runs/<RUN>/<CKPT>` + aynı klasördeki `sketch_bank.pt` |
| Çıktı | Colab'de `/content/gallery/<RUN>/`; GitHub `results/<RUN>/gallery` |
| GPU | herhangi (CPU'da yavaş) |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Modeli yükle | `RUN`, `CKPT` |
| 2 · Tek ses üret | Form: `LABEL`, `INTENSITY`, `WEIGHT`, `DISTANCE`, `ENVIRONMENT`, `DURATION`, `SPEED`, `TEMPERATURE`, `SEED`, `MC_DROPOUT`, `VARIATIONS`, `REFERENCE_FILE`. Taslak bankasından (ya da senin dosyandan) referans taslağı alır, parametrelerle düzenler, modelle ses üretir. Referans ile düzenlenmiş taslağı görsel olarak, ardından referans sesi ve varyasyonları gösterir. |
| 3 · Galeri | Her kategoriden referans + 2 üretim |
| 4 · Değerlendirme | CLAP ile kategori eşleşmesi (üretilen ve gerçek için), çeşitlilik ve kontrol testleri (şiddet / mesafe / süre / hız taranıp ölçülen seviye, parlaklık ve süreyle Spearman korelasyonu). `metrics.json` yazar. |
| 5 · (Opsiyonel) Stable Audio Open kıyası | Aynı kategorileri metin istemiyle hazır diffusion modeline ürettirip aynı CLAP ölçüsüyle karşılaştırır (HF token + lisans kabulü gerekir) |
| 6 · GitHub'a gönder | Galeri ve metrikler; referanslar yalnız paylaşılabilir lisanslıysa |

Henüz çalıştırılmadı. Benzer çıktıyı 02'nin 7. hücresi (vitrin) üretiyor.

---

## 05_mel_generator: vektör → yeni ses, referanssız (rota v2)

| Hücre | Ne yapar |
|---|---|
| 1 · Ayarlar | `RUN`, `PACK` (48 kHz), `VOCODER_RUN` (02'nin `MODE="vocoder"` koşusu), `MAX_HOURS`, GPU'ya göre model genişliği (A100: 256 kanal, ~21 M parametre) |
| 2 · Kuru deneme | ~5 dk eğitim: kayıp düşüyor mu, NaN var mı, mel üretici → vocoder zinciri çalışıyor mu |
| 3 · Eğit | kaldığı yerden devam eder (`last.pt`), `MAX_HOURS` dolunca durur; kayıp eğrisi |
| 4 · Sıfırdan üret | kategori vektörleri (her biri 2 tohum), karışımlar (ör. gunshot+thunder), elle vektörler (hiçbir kategoriye ait olmayan rastgele nokta). Her ses için mel resmi, oynatıcı ve "en yakın gerçek kayda uzaklık" |
| 5 · GitHub | `results/<RUN>/`: `index.html`, sesler zip, `scratch.json`, `mel_gen.pt` (fp16), `loss.csv` |

Hiçbir aşamada referans ses kullanılmaz: zamanlama, olay sayısı ve spektrum mel üreticiden, ses ayrıntısı vocoder'dan gelir.

## Hangi sırayla, ne zaman

| Amaç | Çalıştır |
|---|---|
| Rota v2: sıfırdan üretim | 01 (48 kHz paket) → 02 (`MODE="vocoder"`) → 05 |
| İlk kurulum (poc yolu) | 01 → 02 (`MODE="category"`) |
| Base modeli dinlemek | 02 hücre 5 ya da 7 |
| Alt kategori / uzman model | 03 |
| Parametrelerle oynamak, ölçmek, Stable ile kıyaslamak | 04 |
