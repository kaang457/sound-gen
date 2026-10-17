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
| Çıktı | `MyDrive/sound-gen/pack/` → `audio.bin` (tüm sesler), `index.csv` (klip, etiket, bölüm, lisans, atıf), `meta.json` |
| GPU | gerekmez (CPU yeterli) |
| Süre | DCASE + ESC-50 ≈ 10–15 dk; FSD50K ile ≈ 1 saat |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Kaynakları seç | `USE_DCASE`, `USE_ESC50`, `USE_FSD50K` (~18 GB indirme), `USE_FREESOUND` (API anahtarı gerekir), `MAX_PER_LABEL` |
| 2 · İndir ve paketle | Kaynakları indirir, `configs/categories.yaml`'a göre kategorilere eşler, her sesi 22,05 kHz mono'ya çevirir, sessizliği kırpar, en fazla 6 saniyelik parçalara böler, sessiz parçaları atar ve hepsini tek bir dosyada paketler. Kayıtlar train/val/test olarak kaynak dosyaya göre bölünür; bir kaydın parçaları farklı bölümlere düşmez. |
| 3 · Özet ve dinleme | Kategori × bölüm sayım tablosu, kategori başına dakika ve her kategoriden bir örnek |
| 4 · (Opsiyonel) Referansları gönder | Kategori başına bir gerçek sesi `samples/references`'a yollar; yalnızca paylaşılabilir lisanslı klipler |

**Dikkat (ticari kullanım):** Varsayılan kaynaklar ticari model eğitimi için temiz değil. ESC-50 NC lisanslı; DCASE
içinde BBC ve UrbanSound8K kaynaklı klipler var. Ayrıntılar: [RESEARCH.md §7](RESEARCH.md).

---

## 02_train_base: tüm kategorilerden öğrenen base model

| | |
|---|---|
| Girdi | `pack/` |
| Çıktı | `MyDrive/sound-gen/runs/<RUN>/` → `best.pt`, `last.pt`, `step_*.pt`, `sketch_bank.pt`, `log.csv`, `samples/step_*/` |
| GPU | A100 önerilir (L4 / T4 da çalışır, preset otomatik seçilir) |
| Süre | `MAX_HOURS` ile sınırlı (varsayılan 1,5 saat) |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `RUN` (çalışma adı; aynı adla tekrar çalıştırma devam ettirir), `STEPS`, `MAX_HOURS`, `PRESET` (GPU'ya göre model boyutu), `SMOKE_TEST_FIRST` |
| 2 · Smoke test | Sentetik seslerle 1–2 dakikalık uçtan uca deneme: eğitim → fine-tune → üretim. Sadece kodun çalıştığını gösterir. |
| 3 · Eğit | Paketi yerel diske kopyalar ve koşullu WaveFlow'u **sıfırdan** eğitir. Her adımda 8 kesit (~0,74 s) alınır, 20 bantlık taslak çıkarılır, NLL hesaplanıp ağırlıklar güncellenir. 1.000 adımda bir doğrulama + checkpoint, 2.500 adımda bir dinleme örnekleri. Süre dolunca düzgünce durur; tekrar çalıştırılınca kaldığı yerden devam eder. |
| 4 · Kayıp eğrisi | Train ve val NLL grafiği |
| 5 · Dinle | Son örnek turunda her kategori için `_ref` (gerçek) ve `_gen` (aynı taslaktan üretilen) |
| 6 · GitHub'a gönder | Son örnekleri `samples/<RUN>/step_*`'e yollar |
| 7 · Vitrin | `best.pt` ile tek klasörlük demo: her kategori için gerçek ve üretilen, aynı taslakla kategori geçişi, parametre taramaları ve varyasyonlar. `samples/<RUN>/showcase`'e yollar. |

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
| 1 · Ayarlar | `BASE_RUN`, `BASE_CKPT`, `TARGETS` (ör. `footsteps/snow` ya da pakette olan `footsteps`), `STRATEGY`, `STEPS`, `REPLAY` |
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
