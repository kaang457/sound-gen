# Notebook'lar: ne yapıyorlar?

Dört notebook bir zincir oluşturuyor. Ağır kod `soundgen/` paketinde; notebook'lar onu çağırıp sonuçları gösteriyor.

```
01_data ──► pack (Drive) ──► 02_train_base ──► runs/base_v1 (Drive) ──┬──► 03_finetune ──► runs/ft_* (Drive)
                                                                       └──► 04_generate ◄──┘
                                       hepsi ──► samples/ (GitHub; küçük WAV + json)
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
| 1 · Ayarlar | `COMMERCIAL_ONLY` (varsayılan açık), `CATEGORY` (`auto` / `all` / bir etiket), `PACK_NAME`, kaynak seçimi, `MAX_PER_LABEL` |
| 2 · Sayım | Yalnız FSD50K'nın etiket ve lisans dosyalarını indirir (~10 MB). Kategori başına klip sayısını "CC0 + CC BY" ve "tümü" olarak yazar. `auto` ise en çok ticari-temiz klibi olan kategoriyi seçer. |
| 3 · İndir ve paketle | Ticari modda ESC-50 ve DCASE'i atlar, FSD50K'dan yalnız CC0 / CC BY klipleri alır. Sesleri 22,05 kHz mono'ya çevirir, sessizliği kırpar, ≤ 6 s parçalara böler, tek dosyada paketler. Kayıtlar kaynak dosyaya göre train/val/test'e bölünür. Atıf listesini yazar. |
| 4 · Özet ve dinleme | Kategori × bölüm sayım tablosu, dakika, kategori başına 3 örnek ve atıfları |
| 5 · (Opsiyonel) Referansları gönder | Kategori başına bir gerçek sesi atıf dosyasıyla `samples/references/<PACK_NAME>`'e yollar |

**Ticari mod dışı (eski base_v1 paketi):** `COMMERCIAL_ONLY = False`, `USE_DCASE`/`USE_ESC50` açık, `CATEGORY = "all"`,
`PACK_NAME = "pack"`. Bu paket ticari model eğitimi için temiz değil: [RESEARCH.md §7](RESEARCH.md).

---

## 02_train_base: paketteki kategorilerden öğrenen model (tek kategori → uzman model)

| | |
|---|---|
| Girdi | `<PACK>/` |
| Çıktı | `MyDrive/sound-gen/runs/<RUN>/` → `best.pt`, `last.pt`, `step_*.pt`, `sketch_bank.pt`, `log.csv`, `samples/step_*/` |
| GPU | A100 önerilir (L4 / T4 da çalışır, preset otomatik seçilir) |
| Süre | `MAX_HOURS` ile sınırlı (varsayılan 1,5 saat) |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `RUN` (çalışma adı, varsayılan `clean_v1`; aynı adla tekrar çalıştırma devam ettirir), `PACK` (01'in yazdığı paket), `UPSAMPLER` (`interp` = vızıltı düzeltmesi H1, `transposed` = makale / base_v1), `STEPS`, `MAX_HOURS`, `PRESET` (GPU'ya göre model boyutu), `SMOKE_TEST_FIRST` |
| 2 · Smoke test | Sentetik seslerle 1–2 dakikalık uçtan uca deneme: eğitim → fine-tune → üretim. Sadece kodun çalıştığını gösterir. |
| 3 · Eğit | Bitince son örnekleri ve sağlık raporunu otomatik GitHub'a gönderir. Paketi yerel diske kopyalar ve koşullu WaveFlow'u **sıfırdan** eğitir. Her adımda 8 kesit (~0,74 s) alınır, 20 bantlık taslak çıkarılır, NLL hesaplanıp ağırlıklar güncellenir. 1.000 adımda bir doğrulama + checkpoint, 2.500 adımda bir dinleme örnekleri. Süre dolunca düzgünce durur; tekrar çalıştırılınca kaldığı yerden devam eder. |
| 4 · Kayıp eğrisi | Train ve val NLL grafiği |
| 5 · Dinle | Son örnek turunda her kategori için `_ref` (gerçek) ve `_gen` (aynı taslaktan üretilen) |
| 6 · GitHub'a gönder | Son örnekleri `samples/<RUN>/step_*`'e yollar |
| 7 · Vitrin | `best.pt` ile tek klasörlük demo: her kategori için gerçek ve üretilen, aynı taslakla kategori geçişi, parametre taramaları ve varyasyonlar. `samples/<RUN>/showcase`'e yollar. |
| 8 · Sağlık raporu | Eğitimin teşhisi: ölü gradyan / sönen gradyan / derinlik boyunca gradyan oranı, gerçek güncelleme oranı (‖Δw‖/‖w‖), gradyan kırpma sıklığı, ölü kanal, doymuş tanh, kapalı kapı, ölü ReLU, katman ve flow kapatma testiyle katman önemi, overfit / underfit. Her bulgu OK / UYARI / KRİTİK, değer, anlamı ve ne yapılacağıyla. `runs/<RUN>/health/report.html`'e yazar ve GitHub'a yollar. Eski koşularda (base_v1) da çalışır. |

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
| 5 · GitHub'a gönder | Karşılaştırmayı `samples/ft_*`'e yollar |

Henüz çalıştırılmadı.

---

## 04_generate: parametrik üretim, değerlendirme, kıyas

| | |
|---|---|
| Girdi | `runs/<RUN>/<CKPT>` + aynı klasördeki `sketch_bank.pt` |
| Çıktı | Colab'de `/content/gallery/<RUN>/`; GitHub `samples/<RUN>/gallery` |
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

## Hangi sırayla, ne zaman

| Amaç | Çalıştır |
|---|---|
| İlk kurulum | 01 → 02 (hücre 2 smoke test dahil) |
| Base modeli dinlemek | 02 hücre 5 ya da 7 |
| Alt kategori / uzman model | 03 |
| Parametrelerle oynamak, ölçmek, Stable ile kıyaslamak | 04 |
