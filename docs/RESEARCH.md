# Araştırma: mel spektrogramdan ses üretimi ve bizim için anlamı

*2026-10-17. Pazartesi'deki büyük eğitimden önce. Bağlam için: [PROCESS.md](PROCESS.md).*

## 0. Önce: şu an sıfırdan mı eğitiyoruz?

**Evet.** `base_v1` rastgele ağırlıklarla başladı. Hazır bir model kullanmadık; aldığımız tek şey WaveFlow'un
**tarifi** (mimari). 90 dakikada 11,8k adım ve ~8 saat veri ile eğitildi.

Literatürle kıyaslayınca bu çok az kalıyor:

| Model | Eğitim adımı |
|---|---|
| WaveFlow (konuşma) | 1M |
| HiFi-GAN | ~2,5M |
| BigVGAN-v2 | 5M (100× daha büyük veri) |
| base_v1 | 11,8k |

"Sıfırdan vokoder" bu bütçeyle gerçekçi değil. Bu araştırmanın ana sonucu da bu: **sesi dalga formuna çevirme işini hazır,
büyük bir modele bırakmak**.

---

## 1. İki ayrı problem

Mel'den ses üreten bütün güncel sistemler işi ikiye bölüyor:

```
(A) kategori + parametreler ──► MEL SPEKTROGRAM ──(B) vokoder──► dalga formu
    "ne duyulacak"                                "nasıl duyulacak"
```

- **(B) Vokoder:** Mel'i dalga formuna çeviriyor. Bu kısım evrensel; her sese aynı fizik uygulanıyor. Büyük verilerle bir kez eğitilmiş modeller hazır bulunuyor.
- **(A) Mel üretici:** Kategoriye ve parametrelere göre mel'i üretiyor. Bizim veriye, kategorilere ve kontrollere özgü kısım bu. Ayrıca küçük bir problem: 4 saniyelik ses 88.000 örnek, ama 80 bantlık mel'i sadece 80 × 345 boyutunda.

`base_v1` ikisini birden, sıfırdan ve tek ağda öğrenmeye çalıştı. Andreu & Villanueva'nın tek kategoride çalışan yöntemi
bu; 15 kategoriye ve bizim bütçemize ölçeklenmedi.

---

## 2. Vokoder yöntemleri (B): ne denenmiş

| Aile | Örnekler (yıl) | Fikir | Kalite / hız | Hazır model | SFX için |
|---|---|---|---|---|---|
| Sinyal işleme | Griffin-Lim | fazı yinelemeli tahmin | düşük, "metalik" | gerekmez | sadece taban çizgisi |
| Otoregresif | WaveNet (2016), WaveRNN | örnek örnek üretim | çok iyi / çok yavaş | eski | pratik değil |
| Flow | WaveGlow (2019, 88M), FloWaveNet, **WaveFlow (2020, 5,9M)** | tersinir dönüşüm, tam olasılık | iyi / hızlı; sabit frekanslı ton artefaktı bilinen sorun | sadece konuşma | bizim v1 bu, sıfırdan |
| **GAN** | MelGAN (2019), Parallel WaveGAN, **HiFi-GAN (2020)**, UnivNet (2021), **BigVGAN (2023) / v2 (2024)** | üretici + ayırt edici; tek geçişte dalga formu | çok iyi / çok hızlı | **evet**; BigVGAN-v2 genel ses | **güncel standart** |
| Fourier tabanlı GAN | iSTFTNet, Vocos (2023) | genlik + faz tahmini, ters STFT | iyi / en hızlı | Vocos: sadece konuşma (LibriTTS) | ince ayar gerekir |
| Diffusion | DiffWave, WaveGrad (2020), PriorGrad, FastDiff | gürültüden yinelemeli arındırma | çok iyi / yavaş | sınırlı | yavaş |
| Flow matching | WaveFM (2025) | diffusion'ın düz yollu versiyonu, az adım | çok iyi / orta | yeni | izlenecek |
| Nöral codec | EnCodec, DAC | mel yerine öğrenilmiş latent | çok iyi | evet | mel arayüzü yok |

**Öne çıkan: BigVGAN-v2** (NVIDIA, MIT lisansı)
- **Veri:** Konuşma, **gündelik çevresel sesler** ve enstrümanlar; v1'den 100 kat büyük veri, 5M adım.
- **Bizim için hazır sürüm:** `bigvgan_v2_22khz_80band_256x` (112M parametre). Mel ayarları 22050 Hz, n_fft 1024, hop 256, 80 bant; **bizim veri hattımızla aynı çerçeve**.
- **Genelleme:** v1 sadece konuşmayla eğitilmesine rağmen alkış, müzik ve gürültülü kayıtları sıfır-atış üretebiliyordu. Bunu periyodik "Snake" aktivasyonu ve anti-aliasing'e bağlıyorlar.
- **Fine-tune:** Resmî olarak destekleniyor. Üretici ve ayırt edici checkpoint'leriyle eğitime devam ediliyor; varsayılan batch 4, tek A100'e sığıyor.

---

## 3. Mel üretimi (A) ve ses efektleri literatürü

| Çalışma | (A) mel üretici | (B) vokoder | Not |
|---|---|---|---|
| DCASE 2023 Task 7 temel sistemi | VQ-VAE + PixelSNAIL | HiFi-GAN | modüller ayrı eğitiliyor |
| **DCASE 2023 Task 7 kazananları** | **diffusion → spektrogram** | **HiFi-GAN** | organizatörlerin özeti; iki track'te de en iyiler bu yapıda |
| Yuan vd. (DCASE 2023) | AudioLDM: 2,2M klipte ön eğitim → DCASE verisine **fine-tune** | HiFi-GAN (ön eğitimli) | FAD 4,77 (temel sistem 9,7) |
| Andreu & Villanueva (AIIDE 2022) | yok; 20 bantlık mel'i **elle/örnekten** veriyor | WaveFlow (sıfırdan, tek kategori) | bizim v1'in kaynağı |
| Comunità vd. (2021) | koşullu WaveGAN | HiFi-GAN hibrit | ayak sesi, yüzeye göre |
| Sketch2Sound (2025) | DiT + zaman değişkenli kontroller (yükseklik, parlaklık, perde) | latent codec | kontrol fikri bizimkiyle aynı |

**Ortak desen:** Başarılı Foley sistemleri **(A)'yı kendi verisiyle eğitiyor, (B) için hazır/ön eğitimli vokoder
kullanıyor.** En iyi sonuç ise (A)'yı da büyük veride ön eğitip fine-tune edenlerden geliyor (Yuan vd.).

DCASE organizatörlerinden bir uyarı: FAD gibi otomatik metrikler dinleme sıralamasıyla zayıf korele (Spearman 0,24–0,52).
Dinleme şart; bunu biz de yaşadık.

---

## 4. Bizim için öneri

### Önerilen mimari: v3

```
kategori + parametreler
        │
        ▼
(A) MEL ÜRETİCİ (bizim eğittiğimiz, küçük)        ← transfer learning: base → kategori → alt kategori
    girdi: 20 bantlık taslak (parametrelerle düzenlenmiş) + kategori
    çıktı: 80 bantlık mel
    model: koşullu flow matching / diffusion U-Net (80 × T görüntü)
        │
        ▼
(B) BigVGAN-v2 22 kHz (hazır, NVIDIA)             ← transfer learning: istenirse SFX verimizle kısa fine-tune
        │
        ▼
      ses
```

- **Neden daha iyi:** Ses kalitesini 5M adımla eğitilmiş bir modele bırakıyoruz. Bizim compute'umuz sadece "ne duyulacak" problemine gidiyor ve o problem 250 kat küçük (88.000 örnek → 80 × 345 mel).
- **Parametrik kontrol korunuyor:** Kontroller zaten taslak/mel üzerinde çalışıyor (`controls.py` bant sayısından bağımsız).
- **Varyasyon nereden:** (A) üretken bir model olduğu için her tohumda farklı ayrıntılı mel üretir. (B) deterministik.
- **Zincir hatası:** (B) neredeyse kayıpsız bir çevirici olduğu için (A)'nın hatasını büyütmez; ne üretilirse onu seslendirir.

### Pazartesi sırası

1. **10 dk, eğitim yok (kritik kontrol):** Gerçek seslerimizin 80 bantlık mel'ini BigVGAN-v2'ye verip yeniden sentezle.
   - Gerçek sese yakın çıkıyorsa (B) çözülmüş demektir.
   - Bu aynı zamanda tavan kaliteyi gösterir: (A) mükemmel olsa bile ulaşılabilecek en iyi ses.
2. **Eğitimsiz demo seçeneği:** Referansın tam mel'i + parametrik düzenleme → BigVGAN. Hemen yüksek kaliteli ve kontrol edilebilir ses verir, ama yeni ses üretmez (varyasyon sadece referans seçiminden).
3. **Büyük eğitim:** (A) mel üreticiyi eğit. Önce 7 DCASE kategorisi.
4. **Opsiyonel:** BigVGAN'ı SFX verimizle birkaç bin adım fine-tune et. Ancak 1. adımda kalite yetersizse gerekli.

### Alternatifler (neden ikinci sırada)

- **AudioLDM / Stable Audio Open fine-tune:** En yüksek kalite potansiyeli var, ama ağır. Önceki Colab serisinde sayısal kontrol bağlamak başarısız oldu (PROCESS.md §0.b) ve sürekli parametre kontrolü doğal değil.
- **WaveFlow'a devam (v2 hipotezleri):** PROCESS.md §9'daki testler hâlâ geçerli. Ama sıfırdan vokoder eğitme maliyeti değişmiyor.

---

## Kaynaklar
- Ping vd., *WaveFlow*, ICML 2020 — arXiv:1912.01219
- Andreu & Villanueva Aylagas, *Neural Synthesis of Sound Effects Using Flow-Based Deep Generative Models*, AIIDE 2022
- Lee vd., *BigVGAN: A Universal Neural Vocoder with Large-Scale Training*, ICLR 2023 — https://arxiv.org/abs/2206.04658, https://research.nvidia.com/labs/adlr/projects/bigvgan/
- BigVGAN-v2 model kartı — https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x ; NVIDIA blog — https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/ ; kod — https://github.com/NVIDIA/BigVGAN
- Siuzdak, *Vocos*, 2023 — https://arxiv.org/abs/2306.00814, https://github.com/charactr-platform/vocos
- Choi vd., *Foley Sound Synthesis at the DCASE 2023 Challenge* — https://arxiv.org/abs/2304.12521
- Yuan vd., *Latent Diffusion Model Based Foley Sound Generation System for DCASE 2023 Task 7* — https://arxiv.org/abs/2305.15905
- *WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching*, 2025 — https://arxiv.org/abs/2503.16689
- Kong vd., *HiFi-GAN*, NeurIPS 2020; Kong vd., *DiffWave*, ICLR 2021; Prenger vd., *WaveGlow*, ICASSP 2019
- Comunità, Phan, Reiss, *Neural Synthesis of Footsteps Sound Effects with GANs*, 2021
- Flores García vd., *Sketch2Sound*, ICASSP 2025
