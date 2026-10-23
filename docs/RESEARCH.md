> **23 Ekim notu:** Bu belge araştırma notlarıdır; bazı bölümler (base model + transfer learning planı) rota v2 ile aşıldı.
> Güncel mimari: [README](../README.md) ve [PROCESS.md §11](PROCESS.md).

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

## 5. Ticari kullanım: lisanslar ve hibrit mimari

*Hukuki tavsiye değildir; ticari kullanımdan önce lisans metinlerinin aslı okunmalı.*

### Bileşenler

| Bileşen | Lisans | Ticari çıktı |
|---|---|---|
| BigVGAN-v2 (NVIDIA) | MIT | ✅ |
| Stable Audio 3 Small (SFX) / Stable Audio Open | Stability AI Community License | ✅ yıllık geliri < 1M USD olanlar için, kayıt ile; çıktılar kullanıcının |
| AudioLDM / AudioLDM2 | CC BY-NC-SA 4.0 | ❌ ticari kullanım yok |
| Bizim kod | bizim | ✅ |
| Bizim eğittiğimiz ağırlıklar (WaveFlow, mel üretici) | eğitim verisinin lisansına bağlı | ancak veri ticari kullanıma uygunsa |

### Eğitim verisi

| Kaynak | Ticari amaçla model eğitimi |
|---|---|
| DCASE 2023 T7 | CC BY 4.0 kısmı ✅ (atıf); **BBC kaynaklı klipler ❌** (yalnız araştırma) |
| ESC-50 | ❌ CC BY-NC |
| FSD50K | klip bazında: yalnız CC0 / CC BY olanlar ✅ (atıf) |
| Freesound API | CC0 / CC BY filtresiyle ✅ (atıf) |
| BBC Sound Effects (RemArc) | ❌ ticari değil |
| Sonniss GDC paketleri | ❌ **lisans AI eğitimini yasaklıyor** |
| Kendi kayıtların | ✅ |

`base_v1` ESC-50 ve BBC kaynaklı DCASE klipleriyle eğitildi, yani **ticari kullanıma uygun değil**. Araştırma ve demo için uygun.

### Hibrit mimari: hazır model + bizim katmanlarımız

```
1 · KAYNAK ("ne?")          Stable Audio 3 Small SFX  │  lisanslı ses kütüphanesi  │  kullanıcının sesi
                            (kategori → sabit, iç istem; kullanıcı istem yazmaz)
        │ ses
        ▼
2 · KONTROL (bizim)         mel/taslak çıkar → intensity, weight, distance, environment, duration, speed
        │ düzenlenmiş mel
        ▼
3 · SESLENDİRME ("nasıl?")  BigVGAN-v2 (varsayılan, deterministik)  │  bizim WaveFlow (aynı taslaktan farklı varyasyonlar)
        │
        ▼
      çıktı
```

- **1. katman:** Elimizde verisi olmayan kategorileri (patlama, at, karda adım) de kapsar ve her seferinde yeni içerik üretir.
- **2. katman:** Tamamen bizim, deterministik; lisans sorunu yok. Parametrik kontrolün tamamı burada.
- **3. katman:** Varsayılan olarak BigVGAN-v2. Bizim WaveFlow'un özgün rolü **varyasyon**: aynı taslaktan farklı ayrıntılar (Andreu & Villanueva). Ticari kullanım için yalnız temiz veriyle yeniden eğitilmesi gerekir.
- **İleride:** Öğrenilmiş mel üretici (§4), 1. katmanın yerini kısmen alabilir.
- **Açık soru:** Stable Audio çıktılarıyla kendi modellerimizi eğitmek. Hem ticari temiz veri sorununu çözer hem de kapsamı genişletir. Ama Community License'ın "çıktılarla başka model eğitme" maddesi kontrol edilmeden yapılmamalı.

---

## 6. Kendi WaveFlow'umuzu Stable Audio seviyesine yaklaştırmak

**Ölçek farkı** (Stable Audio Open makalesi, arXiv:2407.14358):

| | Stable Audio Open | base_v1 |
|---|---|---|
| Veri | 486.492 kayıt, **7.300 saat** (Freesound + FMA, CC0 / CC BY) | ~8 saat |
| Model | ~1,32 milyar parametre | ~22 milyon |
| Compute (yalnız DiT) | 338 saat × 64 A100 ≈ **21.600 A100-saat** | 1,5 A100-saat |

**Sonuç:** Genel amaçlı (her istemi anlayan) Stable seviyesi bu bütçeyle mümkün değil. Ama **dar alanda**, yani
seçilmiş kategorilerde ve taslak koşulluyken, eğitim setine yakın kalite mümkün. Andreu & Villanueva tek kategoride bunu
gösterdi.

**Kaldıraçlar (etki sırasıyla):**
1. **Veri:** Kategori başına saatlerce temiz kayıt. Stable Audio Open'ın kendisi Freesound CC0/CC BY ile eğitildi; aynı kaynağa Freesound API ile erişebiliriz.
2. **Yapısal düzeltmeler:** PROCESS.md §9 H1–H2 (taslağı açma şekli, tam hassasiyet).
3. **Uzmanlaşma:** Tek dev model yerine base → kategori uzmanları (transfer learning).
4. **Daha uzun eğitim:** 100k+ adım; 64 kanal ile tek A100'de birkaç saat.
5. **Taslak çözünürlüğü 20 → 30 bant:** kalite ↑, çeşitlilik ↓ (makaledeki denge).
6. **Adversarial ince ayar:** HiFi-GAN ayırt edicileriyle; algısal netliği artırır.
7. **Stable'dan damıtma** (çıktılarla eğitim): en güçlü kaldıraç. Ama lisansın "çıktılarla model eğitimi" maddesi kontrol edilmeden yapılmamalı.

**Kullanılamayan veri:** Sonniss GDC paketleri (2026 dahil). Lisans açıkça diyor ki: "expressly prohibited from using any
sound effects … for the purpose of training artificial intelligence technologies" (https://sonniss.com/gdc-bundle-license/).
Telifsiz (royalty-free) olması açık kaynak olduğu anlamına gelmiyor; oyunlarda kullanılabilir, model eğitiminde kullanılamaz.

---

## 7. Ticari model eğitimi için veri kaynakları

*Lisans metinlerinin aslı, kullanmadan önce okunmalı. Hukuki tavsiye değildir.*

| Durum | Kaynak | Not |
|---|---|---|
| ✅ temiz | **Freesound API**, CC0 + CC BY filtresi | Stable Audio Open'ın tüm Freesound verisi (6.330 saat) bu lisanslardan; CC BY için atıf listesi tutulmalı |
| ✅ temiz | **Kendi kayıtların** | en güvenlisi; ayak sesi, yüzey, zıplama gibi Foley kayıtları için ideal |
| ✅ temiz | **Kenney ses paketleri** (OpenGameArt) | CC0, oyun tarzı kısa efektler (UI, RPG, darbe), az miktarda |
| ✅ temiz | **Greatest Hits** (vis-data) | CC BY 4.0; bagetle darbe/sürtünme, çok sayıda malzeme |
| ⚠️ filtreli | **FSD50K** | klip bazında CC0 / CC BY / CC BY-NC / Sampling+; metadata'dan **yalnız CC0 + CC BY** |
| ⚠️ filtreli | **DCASE 2023 T7** | klipler UrbanSound8K (NC ❌), FSD50K (klip bazında) ve BBC (❌) kaynaklı; DevMeta.csv'ye göre ayrılmalı |
| ⚠️ filtreli | **Clotho** | ses dosyaları Freesound lisanslarında (çoğu CC BY); açıklama metinleri NC, ama onlara ihtiyacımız yok |
| ⚠️ filtreli | **OpenGameArt** (genel) | lisans karışık (CC0, CC BY, CC BY-SA, GPL); CC BY-SA'nın "aynı lisansla paylaş" şartı model ağırlıkları için belirsiz |
| ❓ belirsiz | Pixabay | Pixabay kendi AI eğitimi için kullanıyor; indiren kullanıcının model eğitimi için açık izin bulunamadı |
| ❓ belirsiz | Zapsplat | lisans AI eğitimine değinmiyor; tek başına dağıtım yasağı var |
| ❌ hayır | ESC-50, UrbanSound8K | CC BY-NC |
| ❌ hayır | AudioSet, VGGSound | YouTube videoları; ses lisansı yok |
| ❌ hayır | BBC Sound Effects | ticari değil (RemArc) |
| ❌ hayır | Sonniss GDC | lisans AI eğitimini açıkça yasaklıyor |
| 💰 ücretli | **Pro Sound Effects** AI eğitim veri seti | 1,27M ses, 5.000+ saat, "rights-cleared"; Sony AI da lisanslı kütüphanelerle (PSE, BOOM) eğitmiş |

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
- Stable Audio lisansları — https://stability.ai/license ; https://stability.ai/explainers/stable-audio-vs-competitors-licensing-export-rights-and-self-hosting-compared
- AudioLDM lisansı — https://osai-index.eu/model/audioldm
- Sonniss GDC lisansı (AI eğitimi yasak) — https://sonniss.com/gdc-bundle-license/
- Evans vd., *Stable Audio Open*, 2024 — https://arxiv.org/abs/2407.14358
- Clotho lisansı — https://zenodo.org/records/4783391 ; Pixabay SSS — https://pixabay.com/service/faq/ ; Zapsplat — https://licenseorg.com/guide/music-audio/zapsplat
- Pro Sound Effects / Professional Sound Alliance — https://www.musicbusinessworldwide.com/sound-designers-and-sfx-libraries-launch-professional-sound-alliance-to-fight-ai-scraping-now-sound-will-have-protection-of-our-own/
