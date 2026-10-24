# Mimari, yol haritası ve veri

Kısa, paylaşılabilir sürüm (Claude Docs): [Hedef Mimari ve Veri Satın Alma Raporu](https://claude.ai/artifact/LzG1mmKkpw1a5qFUhyjpYM).

## Özet

Sıkıştırılmış ses latentleri üzerinde çalışan, flow matching ile eğitilen **tek bir çok-koşullu transformer**
(MMAudio mimarisi). Önünde hazır, donuk kodlayıcılar (metin; ileride video), arkasında hazır bir çözücü (mel-VAE +
BigVGAN-v2). Sadece üreticiyi biz eğitiyoruz. Video-to-audio ayrı bir model değil, aynı üreticinin boş duran
video girişinin doldurulması.

## Parçalar

| Parça | Görevi | Kimin | Faz |
|---|---|---|---|
| Üretici (flow matching MM-DiT, 157 M – 1 B parametre) | ses latentini istekten (ve videodan) yaratır; 25 adımda, 8 s ses ~1,3 s | **bizim, eğitilir** | 1 metin, 2 video |
| mel-VAE + BigVGAN-v2 | sesi 345 × 40'lık latente sıkıştırır, geri açar; 44,1 kHz | hazır | 1 |
| CLIP metin kodlayıcı (DFN5B ViT-H/14) | isteği 77 × 1024 token'a çevirir | hazır, donuk | 1 |
| CLIP görüntü (8 fps) + Synchformer (24 fps) | ne görünüyor + olayların tam zamanı | hazır, donuk | 2 |
| Sahne planlayıcı (görüntü-dil modeli) | videodaki ses kaynaklarını ve zaman aralıklarını listeler | hazır | 2 |
| 3 → 8 aday | 3 başlangıç gürültüsü + aralarında 5 ara değer (slerp; üçlü merkez (a+b+c)/√3) | bizim | 1 |
| Efekt rafı | şiddet, uzaklık, ortam (sentetik IR konvolüsyonu), stereo konum, süre | bizim (DSP) | 1 |
| Sahne miksi | kaynak başına iz; beğenilmeyen iz istekle yeniden üretilir | bizim | 2 |
| TTS | konuşma; ayrı model, ayrı veri | ayrı iş kolu | 3 (gerekirse) |

## Üreticinin içi (MMAudio)

- Girdi: ses latenti, saniyede ~43 token (8 s = 345 token), her biri 40 sayı → doğrusal katmanla iç boyuta.
- Koşullar token dizisi: metin 77, video içerik 64 (8 fps × 8 s), video zamanlama 192 (Synchformer).
- Zaman (t) + metin ve videonun özet vektörleri → **adaLN** ile her bloğu ölçekler / kaydırır (FiLM benzeri).
- **Ortak bloklar (MM-DiT)**: ses, metin, video kendi ağırlıklarıyla, ama attention hep beraber; ses ve video
  aynı zaman ekseninde hizalı RoPE kullanır (senkron).
- **Sadece ses blokları**: koşul bilgisi sese geçtikten sonra rafine eder; MLP'lerinde 3'lük 1-D konvolüsyon.
- Çıkış: her token için hız vektörü. Kayıp: (x₁ − x₀) ile tahmin arasındaki kare fark; üretimde Euler, 25 adım,
  classifier-free guidance 4,5.

| Model | İç boyut | Ortak / ses blokları | Parametre | MMAudio eğitimi |
|---|---|---|---|---|
| small_44k | 448 | 4 / 8 | 157 M | 52 H100-saat |
| medium_44k | 896 | 4 / 8 | 621 M | 168 H100-saat |
| large_44k | 896 | 7 / 14 | 1,03 B | 304 H100-saat |

## Neden bu mimari

| Sistem | Yapı | Veri | Lisans |
|---|---|---|---|
| MMAudio (Sony, CVPR 2025) | mel-VAE + flow matching MM-DiT + BigVGAN-v2 | ~8.300 sa (VGGSound, WavCaps, AudioCaps, Clotho) | kod MIT, ağırlık CC BY-NC |
| HunyuanVideo-Foley (Tencent, 2025) | 48 kHz VAE + MM-DiT | ~100 bin sa | AB / İngiltere / G. Kore hariç; çıktılar da |
| Stable Audio Open (2024) | VAE + T5 + DiT | ~7.300 sa CC | Stability Community License |
| Woosh (Sony AI, 2026) | metin + video → ses; ticari yönlü sürümü Pro Sound Effects + BOOM ile eğitildi | açıklanmadı | açık sürüm ticari değil |

- Kodu MIT olduğu için kendi verimizle sıfırdan eğittiğimiz ağırlıklar bizim olur.
- Asıl maliyet veri: büyük model bile ~300 H100-saat (birkaç bin dolar).
- Hazır video modelleri ticari ürüne doğrudan girmiyor; kendi modelimizin gerekçesi bu.

## Yol haritası (tahmini)

1. **Faz 0 · hazırlık (2–3 hafta):** lisans teklifleri; altyapı (bu repo).
2. **Faz 1 · metin → SFX (4–8 hafta):** lisanslı veriyle üretici, 8 adaylı stüdyo. Kapı: ezber yok, CLAP doğruluğu gerçeğe yakın, dinleme onayı.
3. **Faz 2 · video → ses (2–3 ay):** video koşulu, sahne planlayıcı, oyun motorundan kaynak başına izli yakalama verisi. Kapı: senkron ölçümü.
4. **Faz 3 · konuşma (gerekirse):** ayrı TTS.

## Veri

| Havuz | Hedef | Kaynak |
|---|---|---|
| Ses efektleri (tek kaynaklı, açıklamalı) | 1.000–3.000 sa, büyüme 5.000+ | Pro Sound Effects (5.000+ sa AI seti), BOOM, Shutterstock — fiyat teklifle |
| Video + ses | 300–1.000 sa | oyun motoru yakalamaları (kaynak başına iz), Troveo gameplay verisi |
| Konuşma | 1.000+ sa / dil | Appen, Shaip, LXT, Defined.ai; yayımlanmış medyan ~118 $/sa |

Compute iki faz için ~5–36 bin $; veri fiyatları yayımlanmıyor (yayımlanmış medyanlarla 5.000 sa ≈ 210–590 bin $
mertebesi, sadece referans). Kaynaklar ve hesaplar Docs raporunda.
