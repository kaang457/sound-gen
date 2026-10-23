# Notebook'lar: ne yapıyorlar?

Rota v3: **01 → 03 → 04**. Ağır kod `soundgen/` paketinde; notebook'lar onu çağırıp sonuçları gösteriyor.

```
01_data ──► pack48_poc (Drive) ──► 03_mel_generator ──► runs/melgen_v1 (Drive) ──► 04_studio (GUI, paylaşım linki)
                                        │  CLAP vektörleri, eğitim, sıfırdan üretim + test
                                        ▼
                         results/melgen_v1/ (GitHub: sayfa, sesler, ölçümler, model)
02_vocoder (opsiyonel): kendi WaveFlow vocoder'ımız, BigVGAN-v2 ile kıyas için
```

Her notebook'un **hücre 0'ı aynı**: repodaki son kodu çeker, paketi kurar ve Drive'ı bağlar.
Eski poc yolu `notebooks/legacy/` altında; rota v3'te kullanılmıyor.

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

## 02_vocoder (opsiyonel): kendi WaveFlow vocoder'ımız

Ana vocoder hazır BigVGAN-v2; bu notebook WaveFlow'u aynı mel tanımıyla (44,1 kHz, 128 bant, hop 512) evrensel
vocoder olarak eğitir. Sonuç `results/<RUN>/showcase.html`'de gerçek kayıtla yan yana. 03 ve 04'te `VOCODER` olarak
`runs/<RUN>/best.pt` verilirse BigVGAN'ın yerine geçer.

---

## 03_mel_generator: bizim modelimiz, sıfırdan üretim ve test

| Hücre | Ne yapar |
|---|---|
| 1 · Ayarlar | `RUN`, `PACK`, `MAX_HOURS` (varsayılan 4), GPU'ya göre model genişliği (A100: 256 kanal) |
| 2 · CLAP + kuru deneme | paketteki her klibin CLAP ses vektörü (bir kez, ~15–30 dk, Drive'a kaydedilir); ~5 dk eğitim: kayıp düşüyor mu, NaN var mı, istek → mel → BigVGAN zinciri çalışıyor mu |
| 3 · Eğit + otomatik yayın | kaldığı yerden devam eder; süre dolunca test setini üretir, ölçer ve `results/<RUN>/`'a gönderir (bilgisayar kapalı olabilir) |
| 4 · (Elle) tekrar | farklı istekler / kategoriler / karışımlar / istek gücüyle aynı test ve gönderim |
| 5 · Kayıp eğrisi | diffusion kaybı |

GitHub'a giden: `index.html` (her ses: mel, oynatıcı, en yakın eğitim kaydına uzaklık, CLAP sonucu), seslerin zip'i,
`scratch.json`, `clap.json`, `loss.csv`, `mel_gen.pt` (fp16).

---

## 04_studio: GUI

Tek hücre: gradio'yu kurar, modeli (Drive ya da GitHub'dan) ve BigVGAN'ı yükler, herkese açık geçici bir bağlantı verir.

| Bölüm | Ne yapar |
|---|---|
| Üret | istek, opsiyonel ikinci istek + payı, uzunluk (1–5 s), istek gücü, olay sayısı, tohum |
| Sesler | 4 varyasyon + 6 karışım (V1×V2 …: iki varyasyonun başlangıç gürültüsünün ortasından üretilen yeni ses) |
| İşle | şiddet (dB), uzaklık, ortam (kuru / oda / salon / mağara / açık alan), yankı miktarı, stereo konum, süre çarpanı (perde korunur), kırpma, fade |
| İndir | seçilen ses mono + stereo WAV (24 bit) ya da bütün sesler zip |

---

## Hangi sırayla, ne zaman

| Amaç | Çalıştır |
|---|---|
| Sıfırdan üretim (ilk kez) | 01 → 03 |
| İş sahibine göstermek | 04 |
| Yeni isteklerle test sayfasını yenilemek | 03 hücre 4 |
| Kendi vocoder'ımızı denemek | 02 |
| poc_v1 dönemi modelleriyle çalışmak | `notebooks/legacy/` |
