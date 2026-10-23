# Notebook'lar: ne yapıyorlar?

Üç notebook bir zincir oluşturuyor (rota v2: sıfırdan üretim). Ağır kod `soundgen/` paketinde; notebook'lar onu
çağırıp sonuçları gösteriyor.

```
01_data ──► pack48_poc (Drive, 48 kHz) ──┬──► 02_vocoder ──────► runs/vocoder_v1 ──┐
                                         └──► 03_mel_generator ◄────────────────────┘
                                                    │  vektör → yeni mel → vocoder → ses
                                                    ▼
                                    results/<RUN>/ (GitHub: sayfa, sesler, ölçümler, model)
```

Her notebook'un **hücre 0'ı aynı**: repodaki son kodu çeker, paketi Colab'in hazır torch/numpy kurulumuna dokunmadan
kurar ve Drive'ı bağlar. Repoya yapılan her düzeltme, hücre 0 tekrar çalıştırılınca gelir.

Eski poc yolu (referans taslak + kategori koşullu WaveFlow, transfer learning) `notebooks/legacy/` altında; rota v2'de
kullanılmıyor. Ayrıntıları bu belgenin git geçmişinde ve [PROCESS.md](PROCESS.md) §1–§10'da.

---

## 01_data: veriyi indir, işle, paketle

| | |
|---|---|
| Girdi | internetten veri setleri; isteğe bağlı `MyDrive/sound-gen/custom/<etiket>/` |
| Çıktı | `MyDrive/sound-gen/pack48_poc/` → `audio.bin` (48 kHz int16), `index.csv` (klip, etiket, bölüm, lisans, atıf), `meta.json`, `ATTRIBUTION.csv` |
| GPU | gerekmez |
| Süre / yer | ≈ 1–2 saat (çoğu indirme) · Drive'da ≈ 16–17 GB |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `DATASET = "poc_max"` (varsayılan): lisans filtresiz en büyük veri, DCASE + ESC-50 + FSD50K dev + eval + "other". `commercial`: yalnız CC0 / CC BY (`pack48_full`). Drive boş alanını kontrol eder |
| 2 · Sayım | FSD50K'nın etiket ve lisans dosyalarıyla (~10 MB) kategori başına klip sayısı |
| 3 · İndir ve paketle | 48 kHz mono'ya çevirir, sessizliği kırpar, ≤ 6 s parçalara böler, tek dosyada paketler; kaynak dosyaya göre train/val/test |
| 4 · Özet ve dinleme | kategori × bölüm tablosu, dakika, kategori başına 3 örnek |
| 5 · (Opsiyonel) | paylaşılabilir lisanslı birer referansı GitHub'a gönderir |

---

## 02_vocoder: WaveFlow'u evrensel vocoder olarak eğit

| | |
|---|---|
| Girdi | `pack48_poc` |
| Çıktı | Drive `runs/vocoder_v1/` · GitHub `results/vocoder_v1/` (`showcase.html`: gerçek kayıt ile aynı mel'den vocoder çıktısı yan yana, `health.html`, `model.pt`, README) |
| GPU | A100 önerilir (L4 / T4 preset otomatik) |
| Süre | `MAX_HOURS` (varsayılan 5) + yayın ~10 dk |

| Hücre | Ne yapıyor |
|---|---|
| 1 · Ayarlar | `RUN`, `MODE = "vocoder"` (yalnız mel; `category` eski poc yolu), `PACK`, `MAX_HOURS`, `FIXES`, `PRESET` |
| 2 · Kuru deneme | ~10 dk gerçek veriyle: kayıp düşüyor mu, NaN var mı, yayın ve GitHub gönderimi çalışıyor mu (`results/<RUN>_dry/`) |
| 3 · Eğit ve yayınla | kaldığı yerden devam eder; süre dolunca durur, sonuçları `results/<RUN>/`'a gönderir |
| 4 · Kayıp eğrisi | train / val NLL |
| 5 · Sonuçları aç | README ve vitrin |

22 kHz bir paketle başlatılırsa eğitim ilk dakikada net bir hatayla durur (mel sözleşmesi 48 kHz).

---

## 03_mel_generator: vektör → yeni ses, referanssız (rota v2)

| Hücre | Ne yapar |
|---|---|
| 1 · Ayarlar | `RUN`, `PACK` (48 kHz), `VOCODER_RUN` (02_vocoder koşusu), `MAX_HOURS`, GPU'ya göre model genişliği (A100: 256 kanal, ~21 M parametre) |
| 2 · Kuru deneme | ~5 dk eğitim: kayıp düşüyor mu, NaN var mı, mel üretici → vocoder zinciri çalışıyor mu |
| 3 · Eğit | kaldığı yerden devam eder (`last.pt`), `MAX_HOURS` dolunca durur; kayıp eğrisi |
| 4 · Sıfırdan üret | kategori vektörleri (her biri 2 tohum), karışımlar (ör. gunshot+thunder), elle vektörler (hiçbir kategoriye ait olmayan rastgele nokta). Her ses için mel resmi, oynatıcı ve "en yakın gerçek kayda uzaklık" |
| 5 · GitHub | `results/<RUN>/`: `index.html`, sesler zip, `scratch.json`, `mel_gen.pt` (fp16), `loss.csv` |

Hiçbir aşamada referans ses kullanılmaz: zamanlama, olay sayısı ve spektrum mel üreticiden, ses ayrıntısı vocoder'dan gelir.

---

## Hangi sırayla, ne zaman

| Amaç | Çalıştır |
|---|---|
| Sıfırdan üretim (ilk kez) | 01 → 02 → 03 |
| Daha uzun / büyük vocoder | 02 (aynı `RUN` adıyla devam eder) |
| Yeni vektörlerle üretmek, sayfayı yenilemek | 03 hücre 4–5 |
| poc_v1 dönemi modelleriyle çalışmak | `notebooks/legacy/` |
