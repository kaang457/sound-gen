# Süreç kaydı

Bu belge, parametrik çok kategorili Foley üretimi çalışmasında **ne denendiğini, ne ölçüldüğünü, neyin işe yaramadığını ve
nedenini** kaydeder. Araştırmaya dönüldüğünde başlangıç noktası budur. Tarihler 2026-10. Tüm commit'ler `kaang457/sound-gen`.

---

## 0. Bu repodan önceki denemeler (bağlam)

**a) Eski repo içeriği** (git geçmişi, `9f8fc61` ve öncesi): AudioLDM entegrasyonu, T-Foley çıkarımı, spektral CVAE ile
parametrik kontrol denemeleri. Bu çalışmayla değiştirildi (PR #1). Geri almak için: `git checkout 9f8fc61 -- .`

**b) Colab serisi 02–19 (Stable Audio 3 Small SFX'e sayısal koşul bağlama).** Özet bulgular:
- Metin kodlayıcısı yerine `prompt` yuvasına sayısal token verildi. Çıktılar gürültü, müzik ya da silah benzeri oldu.
- 10. notebook'ta kodlayıcı `lr=1e-5` ile 200 adımda pratikte hiç hareket etmedi.
- `seconds_total` her zaman 5,944 s verildi. Klipler 0,5–1 s olduğu için model 6 saniyeyi doldurmaya çalıştı; çoklu darbe ve müzik çıktılarının olası nedeni bu.
- CFG için koşul dropout'u yoktu; base model CFG=1 ile değerlendirildi.
- Batch 1 ve tek t örneği kullanıldı; loss eğrileri bu yüzden yorumlanamaz.
- Hazır modelin kendisi metinle de tek darbe üretemedi (18. notebook: 36 sesten 0'ı tek darbe).

---

## 1. Yöntem seçimi

**Hedef:** Kategori + parametreler (şiddet, ağırlık, mesafe, ortam, süre, hız) → gerçekçi Foley. Base model → transfer
learning ile kategori ve alt kategori modelleri. Sınırlı veri ve compute.

**Seçilen yaklaşım:**
- **WaveFlow** (Ping vd., ICML 2020): ham ses üzerinde maksimum likelihood ile eğitilen kompakt flow modeli.
- **Düşük çözünürlüklü mel taslak** (Andreu & Villanueva, AIIDE 2022): 20 bantlık koşul. Kontrol yüzeyi olarak da kullanıldı; parametreler taslağı deterministik olarak düzenliyor.
- **Kategori embedding'i:** Makalenin "gelecek çalışma" olarak bıraktığı tek ağda çoklu kategori fikri. Alt kategoriler ebeveynin embedding'inden başlıyor.

**Gerekçeler:** küçük model (6–25M), tek GPU, tam olasılıkla kararlı eğitim, hızlı örnekleme. Makale 10k–50k adımda
eğitim setine yakın kalite raporluyor. Ancak o sonuç **tek kategori** (patlama), 16 kHz ve 64 kanal içindi.

---

## 2. Uygulama ve doğrulama

Modüller: `soundgen/{model,data,train,controls,generate,evaluate,colab,showcase,smoke}.py`, notebook'lar 01–04.

| Kontrol | Sonuç |
|---|---|
| Flow tersinirliği | ters dönüşüm hatası 1,8e-7 |
| Log-det | otomatik Jacobian ile birebir (−2,9020 = −2,9020) |
| Uçtan uca smoke test (CPU, sentetik) | base → fine-tune → parametrik üretim geçti |
| Küçük öğrenme testi (CPU, sentetik) | NLL 100 adımda 0,95 → 0,07 |
| Parametrik kontroller (numpy) | şiddet, mesafe, süre, hız ve ritim beklenen yönde |

Colab sürecinde düzeltilen hatalar: negatif stride (post-processing), Colab torch'unun pip ile bozulması (`--no-deps`),
GitHub push'ta rebase çakışması (staging klonu sıfırlama), boş fine-tune verisinde anlaşılmaz hata, örneklerin 8
kategoriyle sınırlı kalması.

---

## 3. Veri

`01_data`: DCASE 2023 Task 7 (7 kategori, ~600–800 klip) + ESC-50 (40 klip/kategori). FSD50K ve Freesound kullanılmadı.
Base modele giren 15 kategori: birds, car, cough_sneeze, dog_bark, door_knock, fire, footsteps, glass_break, gunshot,
helicopter, keyboard, rain, sea_waves, thunder, wind. ESC-50 kaynaklı kategoriler ~36 eğitim klibiyle 30 sınırının hemen üstündeydi.

---

## 4. base_v1 eğitimi

| Ayar | Değer |
|---|---|
| Model | 8 flow × 8 katman, h=16, 128 kanal, label_dim 64, dropout 0,05 |
| Koşul | 20 bantlık mel taslak, transposed-conv upsampler (makaledeki gibi) |
| Eğitim | batch 8, 16.384 örneklik kesit, AdamW 2e-4, warmup + cosine (15k), EMA 0,999, **bf16**, A100 |
| Süre | 90 dk zaman bütçesi → **11.783 adım** (~0,46 s/adım) |
| Çıktılar | Drive `runs/base_v1/`; GitHub `samples/base_v1/{step_011783,showcase}` |

---

## 5. Sonuç ve teşhis

**Dinleme:** "Hepsi çöp gibi, hepsinde aynı gürültü var." Sadece thunder kulağa hoş geldi.

**Ölçümler** (`showcase`, 15 kategori, gerçek ve üretilen):
1. **Sabit tonlar:** Tüm üretimlerde k × 22050/16 = k × 1378 Hz'de (k = 1…7) çevresinden 3–14 dB yüksek tonlar var; gerçek seslerde yoklar. Satırlar arası standart sapma oranı üretimde 1,1–1,7, gerçekte 1,0–1,1. Bu, 16 örneklik periyodik bir desen demek.
   - Olası kaynak 1: stride-16 transposed-conv upsampler'ın her 16 örnekte tekrar eden koşul deseni.
   - Olası kaynak 2: WaveFlow'un satırlara eşit davranmaması (ilk satır hiçbir şey görmüyor, son satır 15 satır görüyor).
   - Andreu & Villanueva da WaveFlow'da "sabit frekanslı gürültü" bildirip son işlemle temizlemişti.
2. **Seviye takibi yok:** Üretimler referanstan 5–29 dB sessiz (dog_bark −8 → −39 dB, gunshot −19 → −32 dB).
3. **Zarf takibi zayıf:** Dokulu seslerde zarf korelasyonu ~0 (footsteps 0,07, car 0,10). Olay seslerinde kısmen var (dog_bark 0,82, cough_sneeze 0,55).

Yorum: Model taslağı henüz kullanmıyor. Çıktıları büyük ölçüde kategoriye özgü ortalama doku + periyodik artefakt.

---

## 6. Deneme 1: son işlem (`samples/base_v1/postprocess_v1`), kabul edildi

- **Yapılan:** k × 1378 Hz'e tarak notch filtre + üretilen sesin 20 bantlık spektrumunu referans taslağa zorlama (±30 dB).
- **Kontrol:** Aynı işlem beyaz gürültüye uygulandı (`*_noise_baseline.wav`).
- **Ölçüm:** Tonlar −25…−33 dB'ye indi, seviye referansa 0–5 dB yaklaştı, mel korelasyonu ~0,8'den ~0,96'ya çıktı.
- **Dinleme:** "Benzerlik daha fazla ama farklı tonda bir cızırtı var."
- **Karar (19 Ekim):** v1 beğenildi ve tutuldu; v2 reddedildi. v1 artık `Generator.generate(refine="v1")` (kod: `generate.refine_v1`); 02 hücre 7 ve 04'te varsayılan. Not: taslağa eşleme sesi DSP ile taslağa zorluyor; modelin kendi katkısını görmek için `refine="none"` ya da `*_noise_baseline.wav` kontrolüyle kıyasla.
- **Dinleme geri bildirimi (19 Ekim):** v1'de door_knock, dog_bark, cough_sneeze, birds ve thunder "bu eğitim seviyesine göre çok umut verici". Bir sonraki karşılaştırmada aynı beş kategori `*_noise_baseline.wav` ile kıyaslanmalı: fark ne kadar büyükse o kadarı modelin katkısı.

---

## 7. Deneme 2 ("v2"): olumsuz sonuç, geri alındı

**Yapılan** (commit `c1d7a6a`):
- Örnekleme sıcaklığı 0,8 → 0,6
- Taslak eşlemede yükseltme en fazla +12 dB, kısma en fazla −40 dB
- 1378 Hz notch
- Aynı commit'te **hiç çalıştırılmamış** bir yeniden eğitim ayarı: enterpolasyon upsampler, 64 kanal, fp32, 7 DCASE kategorisi

**Sonuç** (`samples/base_v1/rejected/postprocess_v2`): "Thunder bile bozuldu; ya sadece gürültü ya sessizlik." Ölçüm bunu doğruluyor:

| | gerçek | v1 ham | v2post |
|---|---|---|---|
| thunder | −20 dB | −29 dB | **−53 dB**, karelerin %44'ü sessiz |
| gunshot | −21 dB | −50 dB | **−71 dB**, %100 sessiz |
| footsteps | −23 dB | −42 dB | **−70 dB**, %99 sessiz |
| dog_bark | −18 dB | −41 dB | **−56 dB**, %77 sessiz |

**Neden olumsuz sonuçlandı:**
1. **Asimetrik sınır net kısmaya dönüştü.** Model çıktısı zaten 10–30 dB sessizdi. Yükseltme +12 dB'de kesildiği için hedefe ulaşamadı; kısma ise −40 dB'ye kadar serbestti. Taslakla zamanlaması örtüşmeyen her bölge sertçe kısıldı. Sonuç: kapı (gate) gibi açılıp kapanan ses, yani "ya gürültü ya sessizlik".
2. **Düşük sıcaklık seviyeyi daha da düşürdü.** Flow çıktısının genliği başlangıç gürültüsüyle ölçekleniyor. 0,6, zaten sessiz çıktıyı biraz daha kıstı.
3. **Asıl sorun son işlemle çözülemez.** Model taslağı takip etmiyor ve ince yapıyı öğrenmemiş. Son işlem eksik içeriği yaratamaz, sadece var olanı maskeler ya da keser. Thunder'ın v1'deki hoş dokusu, ona zorla bindirilen zarf yüzünden bozuldu.
4. **Değerlendirme hatası (ders).** Deneme 1'i "mel korelasyonu 0,96'ya çıktı" diye başarılı saydım. Ama bu tam olarak zorladığım büyüklüktü; ölçüm döngüseldi (circular). Kaliteyi göstermedi. Gerçek durumu dinleme ve gürültü kontrolü ortaya çıkardı.

Commit `2bbbb5d` ile tamamen geri alındı. Enterpolasyon upsampler / 64 kanal / fp32 hipotezleri **test edilmedi**; aşağıda açık hipotez olarak duruyor.

---

## 8. Dersler

- Bir düzeltmeyi, zorladığı büyüklükle ölçme. Bağımsız metrikler kullan: CLAP kategori doğruluğu, FAD, seviye hatası, ton fazlası. Her zaman dinleme + **modelsiz kontrol** (gürültü taban çizgisi) olsun.
- Son işlem modelin öğrenmediğini telafi edemez. Önce ham çıktının taslağı takip edip etmediğini ölç: seviye farkı ve zarf korelasyonu.
- Makaledeki "10k adım yeter" bulgusu tek kategori, 16 kHz ve 64 kanal içindi. 15 kategori (bazıları 36 klip), 22 kHz ve 128 kanala doğrudan taşınmıyor.
- Compute kısıtlıyken önce **küçük ve hızlı tanı deneyleri**: tek kategori, kısa koşu, ham çıktı ölçümleri. Ancak bunlar geçince büyük eğitim.

---

## 9. Araştırmaya dönülürse: açık hipotezler (öncelik sırasıyla)

| # | Hipotez | Ucuz test | Başarı ölçütü |
|---|---|---|---|
| H1 | 1378 Hz tonları stride-16 transposed-conv upsampler'dan geliyor | Enterpolasyon upsampler ile kısa koşu (ör. tek kategori, 3k adım) vs mevcut | k·sr/h'deki ton fazlası < 2 dB (ham çıktıda, son işlemsiz) |
| H2 | bf16, flow'un örnek düzeyindeki hassasiyeti için yetersiz | Aynı koşu fp32/TF32 ile | daha düşük val NLL, daha az hışırtı |
| H3 | Model az eğitildi / kapasite 15 kategoriye bölündü | Tek kategori (rain ya da gunshot), 64 kanal, makaledeki gibi | seviye farkı < 3 dB, zarf korelasyonu > 0,7 (ham) |
| H4 | Sıcaklık seviye/hışırtı dengesini belirliyor | Ham çıktıda 0,5–1,0 taraması, son işlemsiz | seviye ve ton fazlası eğrileri |
| H5 | Taslak koşulu fazla zayıf bağlanıyor (**base_v1 raporu destekliyor, §9a**) | Koşul gürültü augmentation'ını kapatma; taslak kanalını artırma (20 → 30, makalede en iyi kalite 30) | zarf korelasyonu |

**Alternatif yollar:**
- Hazır diffusion modeliyle zero-shot kıyas (`04_generate`, Stable Audio Open). Kalite tavanını görmek için.
- Diffusion fine-tune, ancak 0.b'deki hatalar düzeltilerek: gerçek `seconds_total`, CFG için koşul dropout'u, batch ≥ 16, metin embedding'inden başlatılmış sayısal token.

---

## 9a. base_v1 sağlık raporu (`samples/base_v1/health/report.html`)

`last.pt` (adım 11.783) ve `log.csv` ile, CPU'da 30 gerçek klip üzerinde (2 × 8192 örneklik kesit, 3 batch).

| Bulgu | Değer | Yorum |
|---|---|---|
| Ölü / sönen gradyan | yok; ilk flow / son flow oranı 1,57 | Gradyan akışı sağlıklı |
| Güncelleme oranı | medyan 1,6·10⁻⁴ (lr 3,4·10⁻⁵'te) | Ağırlıklar hâlâ hareket ediyor |
| Ölü kanal / ölü ReLU | %0 / %2,6 | Sorun yok |
| Gereksiz katman | 0/64 | Her katman katkı veriyor; flow katkısı f2 > f1 > f3 > … > f7 (en az, ΔNLL 0,10) |
| Overfit | yok | Val 12 ölçümün hepsinde düştü (−0,74 → −4,0006) |
| Gradyan kırpma | kayıtların %100'ü (clip 1,0; kırpma öncesi medyan 9,4, en büyük 73) | Kırpma her adımda devrede; ani sıçramalı adımlar orantısız küçülüyor |
| Doymuş tanh | ortalama %8; flow'ların ilk katmanında %18–51, f0.L5–L7'de %32–58 | Gradyanın bir kısmı oradan geçemiyor |
| Kapalı kapı | f0.L7 %65 | Flow 0'ın son katmanı büyük ölçüde kapalı |

**Asıl bulgu: koşul sinyali çok zayıf.** Flow'ların ilk katmanında tanh girişini ikiye ayırınca (4 klip, ek ölçüm):

| | ses yolundan gelen (rms) | taslak + etiketten gelen (rms) | doymuş (yalnız ses yolu) | doymuş (yalnız koşul) |
|---|---|---|---|---|
| flow 0 | 1,81 | 0,20 | %21 | %0 |
| flow 1 | 9,51 | 0,21 | %48 | %0 |
| flow 2 | 8,39 | 0,24 | %51 | %0 |
| flow 3–7 | 2,0–2,9 | 0,20–0,28 | %19–24 | %0 |

- Doygunluk tamamen ses yolundan geliyor. Taslak, ses yolunun **10–45 katı daha zayıf**. Doymuş bir tanh'ta küçük bir ek, çıktıyı neredeyse hiç değiştirmez; yani model taslağı büyük ölçüde duymuyor. Bu, "çıktı taslağın seviyesini ve zarfını takip etmiyor" gözlemiyle (bölüm 5) tutarlı ve **H5'i destekliyor**. Yalnız ilk katmanda ölçüldü; diğer katmanlar için ayrıca bakılmalı.
- Flow 0 sesi yaklaşık e^3,3 ≈ 26 kat büyütüyor (ortalama |log s| = 3,27; girişin rms'i 0,13 → çıkışın rms'i 3,35). Flow 1–2'ye bu büyütülmüş sinyal giriyor ve en büyük ses yolu değerleri orada.

**Önerilen düzeltmeler** (yeni koşu gerektirir; base_v1'e uygulanamaz):
1. Flow'ların girişini normalize etmek: Glow'daki **ActNorm**, ya da ses yolunu öğrenilen bir ölçekle sınırlamak.
2. Koşulu güçlendirmek: taslağı ortalaması 0, varyansı 1 olacak şekilde ölçeklemek ve/veya **FiLM** (koşuldan çarpan + kaydırma) kullanmak.
3. `grad_clip`'i 1,0'dan ~20–30'a çekmek.

---

## 9d. full_v1: bütün ticari-temiz veri + sağlık raporu düzeltmeleri (20 Ekim)

clean_v1 (`pack_clean`) yalnız `car` ile eğitildi: 01'in o zamanki varsayılanı `CATEGORY="auto"` (en çok klibi olan
tek kategori) idi; ayrıca `STEPS=15000` varsayılanı 5 saatlik bütçeyi ~2 saatte bitirmiş olabilir. Yeni varsayılanlar:
- 01: `CATEGORY="all"`, `INCLUDE_OTHER=True` (kategorisiz CC0/CC BY klipler "other" olarak, `MAX_OTHER=8000` parça), paket `pack_full`
- 02: `RUN="full_v1"`, `STEPS=0` (süreden hesaplanır), `MAX_HOURS=5`, `FIXES=True`:
  `cond_norm` (taslak ölçekleme), `actnorm` (flow girişinde ActNorm), dropout 0, `grad_clip` 25
- Sağlık raporu artık "koşul gücü"nü (taslak sinyali / ses yolu) her katmanda ölçüyor.

**Ön test** (sentetik veri, küçük model: 4 flow × 6 katman, 32 kanal; 400 adım, CPU):

| | train NLL | val NLL (EMA) | koşul gücü (medyan) | doymuş tanh |
|---|---|---|---|---|
| düzeltmesiz | −1,61 | 0,800 | 0,16 | %1,8 |
| düzeltmeli, `cond_scale` 0,25 | **−3,28** | **0,676** | **0,70** | %32 |
| düzeltmeli, `cond_scale` 0,5 | −3,20 | 0,729 | 0,38 | %22,5 |

Düzeltmeler koşulu ~4 kat güçlendiriyor ve öğrenmeyi belirgin hızlandırıyor. Doygunluk arttı (ölçek 0,5'te de %22),
yani yalnız taslak ölçeğinden gelmiyor. Varsayılan 0,25 kaldı (en iyi val); full_v1'in sağlık raporunda doygunluk ve
gradyan kırpma sıklığı ayrıca izlenecek. Sentetik ve kısa bir test; gerçek veride doğrulanması gerekiyor.

## 9e. poc_v1: kavram kanıtı için lisans filtresiz en büyük veri (20 Ekim)

Karar: büyük eğitim kavram kanıtı içindir; ticari olmayan veri de dahil. 01 `DATASET="poc_max"`: DCASE 2023 T7 +
ESC-50 + FSD50K'nın tamamı: dev **ve eval** bölümü (+10.231 klip, +27,9 saat; konuşma/müzik hariç), kategorisizler "other" (`MAX_OTHER=16000` parça) → `pack_poc`. Eval indirmesi Zenodo dosya listesine göre yazıldı ve gerçek parçalı zip'lerle (sahte içerik) uçtan uca test edildi.
02 `RUN="poc_v1"`, düzeltmeler açık. **Bu modelin çıktıları ticari kullanılamaz**; ticari yol için `pack_full` duruyor.

## 9f. poc_v1 sonucu (21 Ekim)

81.782 adım, 47,8 saat veri (17 etiket, "other" 16.000 parça), 8×8 flow, 128 kanal, düzeltmeler açık.
- **En iyi val NLL −4,27** (base_v1: −4,00). Val 10.000. adımdan sonra neredeyse düz; 7 kat adım ve ~6 kat veri
  NLL'i yalnız 0,27 iyileştirdi. Bu kurulumla daha fazla compute verimsiz.
- **Sağlık:** ölü/sönen gradyan yok; **koşul gücü 0,51** (base_v1 ~0,02–0,1); kırpma %22; doymuş tanh %12; ölü ReLU %31 (uyarı).
- **Parametreler sese yansıyor** (v1 son işlemi açık): mesafe 0→1 seviye −26→−43 dB, parlaklık 3,3→1,4 kHz;
  ağırlık −1→+1 parlaklık 4,7→1,8 kHz.
- **Dinleme:** beğenilmedi; thunder "bomboş".
- **Sıcaklık hatası (benim, en büyüğü):** üretim sıcaklığı 0,8 (WaveGlow alışkanlığı) bu modelde sesi 7–9 dB kısıyordu;
  1,0'da seviye gerçeğe ~1 dB yakın. Sıcaklık 1,0 ile CPU'da yeniden ölçüldü (`results/poc_v1/noise_test.html`):
  seviye farkı ort. +1,6 dB, spektrum şekli 0,2–1,1 dB, zaman zarfı korelasyonu 0,69–1,00, bant içi ince ayrıntıda model
  16/16 kategoride şekillendirilmiş gürültüden gerçeğe yakın (2,57 / 2,77 dB), 1378 Hz vızıltı yok (ton fazlası ~0 dB → **H1 doğrulandı**).
  v1 son işlemi ince ayrıntıyı hafif bozuyor (2,71); varsayılan artık son işlemsiz, varsayılan sıcaklık 1,0.
- **Hatalar (benim):** `.gitignore` `*.pt`'yi dışladığı için `model.pt` / `sketch_bank.pt` GitHub'a gitmedi;
  vitrin referansı rastgele (ilk) klipti, thunder için seçilen gerçek kayıt zaten çok kısıktı (−38,6 dB);
  dropout 0 iken vitrinde MC dropout varyasyonu anlamsızdı. Üçü düzeltildi.

## 9c. Mimari kararı: tür başına uzman değil, tek ortak model (19 Ekim)

Her ses türü için ayrı bir uzman model, her tür için yeterli veri bulmayı gerektiriyor; bu gerçekçi değil. Karar:
- **Tek ortak WaveFlow:** tüm türler bir modelde, ses türü girdi olarak (etiket embedding'i) veriliyor.
- **Az verili tür:** model donuk, yalnız türün etiket vektörü öğreniliyor (`freeze="embedding_only"`, 03'te < 30 klip için zaten seçiliyor).
- **Alt tür:** etiket vektörü ebeveynden başlıyor (`footsteps/snow` ← `footsteps`), kodda var.
- **Uzman (tam ince ayar):** yalnız verisi bol ve kalitesi kritik türler için isteğe bağlı.

---

## 9b. Sonraki koşu: `clean_v1` (planlandı)

Karar: önce **tek kategoride**, **ticari kullanıma uygun** veriyle bir uzman model. Sonra aynısı diğer kategoriler
için tekrarlanır; ileride bir router istenen kategoriye göre uzman modeli seçer.

| | |
|---|---|
| Veri | FSD50K'dan yalnız CC0 + CC BY klipler (ESC-50 ve DCASE ticari modda atlanır). Kategori: 01'deki sayımda en çok klibi olan (`CATEGORY = "auto"`). Atıf listesi `ATTRIBUTION.csv`. |
| Paket | `MyDrive/sound-gen/pack_clean/` |
| Model | base_v1 ile aynı boyut; `upsampler = "interp"` (H1 testi) |
| Test edilen | H1 (ham çıktıda k·1378 Hz ton fazlası < 2 dB) ve H3 (tek kategoride seviye/zarf takibi) |
| Not | Yalnız upsampler değiştiği için ton fazlası base_v1 ölçümüyle doğrudan kıyaslanabilir; son işlem kullanılmaz. |

---

## 10. Nerede ne var

Eski çıktılar (samples/) 20 Ekim'de `archive` dalına taşındı; bu belgedeki `samples/...` yolları o daldadır.


| Ne | Yer |
|---|---|
| Kod | `soundgen/`, notebook'lar `notebooks/01–04` |
| Eğitim teşhisi | `soundgen/diagnostics.py`; eğitimde `runs/<RUN>/diag.jsonl`, rapor `runs/<RUN>/health/report.html` (02 hücre 8) |
| Veri paketi | Drive `MyDrive/sound-gen/pack/` (DCASE + ESC-50, base_v1) · `pack_clean/` (ticari-temiz, tek kategori, clean_v1) |
| base_v1 checkpoint'leri | Drive `MyDrive/sound-gen/runs/base_v1/` (`best.pt`, `last.pt`, `step_*.pt`, `sketch_bank.pt`, `log.csv`) |
| Dinleme dosyaları | `archive` dalı: `samples/references`, `samples/base_v1/{showcase, step_011783, health, postprocess_v1, rejected/postprocess_v2}` · yeni koşular: `results/<RUN>/` (main) |
| Önemli commit'ler | `da00df1` yeni yapı · `d0dee38` zaman bütçesi + vitrin · `3eede2f` deneme 1 · `c1d7a6a` deneme 2 · `2bbbb5d` geri alma |
| Makaleler | Ping vd. 2020 (WaveFlow, arXiv 1912.01219); Andreu & Villanueva Aylagas 2022 (AIIDE) |

---

## 11. Rota v2: sıfırdan üretim (21 Ekim)

**23 Ekim:** notebook'lar rota v2'ye göre yeniden adlandırıldı: `01_data`, `02_vocoder`, `03_mel_generator`; poc yolu (base model + transfer learning) `notebooks/legacy/`'ye taşındı ve rota v2'de kullanılmıyor.

**Hedef:** oyunlarda kullanılabilecek, referans ses olmadan üretilmiş gerçekçi sesler; endüstriyel ölçeğe büyüyebilecek
bir mimari. Girdi metin olmak zorunda değil: şimdilik bir vektör (kategori vektörü, karışım ya da elle), ileride bir
metin kodlayıcının (CLAP / T5, dondurulmuş) vektörü.

```
 vektör (etiket / karışım / elle; ileride metin kodlayıcı)
   │
 [C] mel üretici: diffusion (1-D U-Net, FiLM, classifier-free guidance, DDIM 50 adım)   soundgen/sketch_model.py
   │      → yeni 128 × 200 mel (2 s)
 [D] kontroller: kural tabanlı (şiddet, uzaklık, ortam, süre, hız, çoklu olay)          soundgen/controls.py
   │
 [E] WaveFlow, evrensel vocoder: yalnız mel'i görür, kategori almaz                     train(vocoder=True)
   │
  ses (48 kHz)                                                                          soundgen/scratch.py zinciri
```

**Kararlar**

| Karar | Neden |
|---|---|
| Mel sözleşmesi: 48 kHz, n_fft 2048, hop 480, 128 bant, 20 Hz–24 kHz, slaney, (dB+100)/100 | Oyun sesi 48 kHz; 128 bant vocoder'ın uydurması gereken ayrıntıyı azaltır (20 bantta model gürültüden yalnız biraz iyiydi: 2,51 / 2,70 dB). Her modül bunu konuşur, değişirse hepsi yeniden eğitilir. `soundgen/audio.py` |
| WaveFlow evrensel vocoder | Sesin ne olduğuna mel üretici karar verir; vocoder yalnız çalar. Etiket gerekmediği için bütün temiz ses (etiketsiz dahil) eğitimde kullanılabilir ve eğitim etiketleri dışındaki mel'leri (ör. sentetik sci-fi) de çalabilir. |
| Modüller ayrı eğitilir, arayüz mel | Vocoder ileride daha hızlısıyla (HiFi-GAN / BigVGAN / Vocos) değiştirilebilir; üst modüller etkilenmez. WaveFlow 16 satırı sırayla ürettiği için endüstriyel ölçekte darboğaz olabilir. |
| Yaratıcılık birinci aşamada | DCASE 2023 Task 7'nin en iyi sistemleri ve AudioLDM / Stable Audio da böyle: mel'i diffusion üretir, vocoder çalar. |
| poc_v1 korunur | Her checkpoint kendi `mel_cfg`'sini taşır; poc_v1 22 kHz / 20 bantla çalışmaya devam eder. 22 kHz pakete 48 kHz eğitim başlatılırsa hata verir. |

**Sıfırdan üretimin testi (03_mel_generator, hücre 4)**
1. *Ezber mi?* `scratch.NoveltyIndex`: üretilen her mel, eğitim kayıtlarının (etiket başına 150'ye kadar, bütün
   etiketler) her kare kaydırmasıyla kıyaslanır (−60 dB üstü hücrelerde ortalama |dB|). Ölçüt: modelin görmediği gerçek
   kayıtların (val) en yakın eğitim kaydına uzaklığı. Birebir kopya 0,0 dB verir (test edildi).
2. *Tanınıyor mu?* CLAP sıfır atış sınıflandırma; üretilen seslerin doğruluğu gerçek val kayıtlarınınkiyle yan yana.
3. *Dinleme:* sayfa (mel resmi + ses + iki ölçü) ve zip.
Uzaklık tek başına kalite göstermez (kötü bir ses de "uzak" çıkar); bu yüzden 1 ve 2 birlikte okunur.

**Veri ihtiyacı (kabaca)**

| Modül | Ne ister | Ne kadar |
|---|---|---|
| Vocoder | etiketsiz temiz ses, her türden (synth tonları dahil) | yüzlerce saat |
| Mel üretici | etiketli / açıklamalı ses | etiket sözlüğü için yüzlerce, serbest metin için binlerce saat |
| Tasarlanmış sesler (sci-fi vb.) | prosedürel sentez (FM, filtreli gürültü, perde kaydırma…) + az sayıda gerçek tasarım | sentez sınırsız, gerçek birkaç yüz–birkaç bin |

**Sonraki adımlar:** (1) 01_data → 48 kHz paket, (2) 02_vocoder, (3) 03_mel_generator + sıfırdan üretim sayfası,
(4) prosedürel sci-fi veri üreteci, (5) dondurulmuş metin kodlayıcı (CLAP) ile vektör yerine kelime.

CPU'da 20 bantlık bir ön deneme (poc_v1 taslak bankasıyla, 6000 adım) yapıldı; render edilmedi, rota 128 banda geçti.
