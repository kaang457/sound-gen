# poc_v1

> **21 Ekim güncellemesi.** İlk vitrin sıcaklık 0,8 ile üretildi; bu modelde sesi 7–9 dB kısıyordu ("boş" ses izlenimi).
> Sıcaklık 1,0 ile seviye gerçeğe oturuyor. Güncel dinleme sayfaları: **`compare_ref_gen.html`** (gerçek ve üretilen) ve
> **`noise_test.html`** (model ne katıyor). Eski vitrin `showcase_eski_sicaklik08.html` olarak duruyor.
>
> Sıcaklık 1,0'da, dinlemede gerçek kaydı yeterince iyi bulunan 8 kategoride (cough_sneeze, dog_bark, door_knock,
> glass_break, gunshot, keyboard, sea_waves, thunder): ortalama seviye farkı +2.1 dB; bant içi ince ayrıntıda model
> 8 kategorinin 8'inde aynı taslağa uydurulmuş gürültüden gerçeğe daha yakın; 1378 Hz vızıltı yok.
> birds, car, explosion, fire, footsteps, helicopter, other, rain ve wind değerlendirmeden çıkarıldı: gerçek kayıtları kötü.

- Adım: **81782** · en iyi doğrulama NLL: **-4.2725** (düşük = iyi)
- Veri: 47.8 saat, 40516 parça, 17 etiket
- Model: 8 flow × 8 katman, 128 kanal, upsampler `interp`, cond_norm True, actnorm True
- Eski vitrin: sıcaklık 0,8, `refine='v1'`

| Dosya | Ne |
|---|---|
| `compare_ref_gen.html` | Her kategori için gerçek ve üretilen (sıcaklık 1,0) |
| `noise_test.html` | Model ne katıyor: model, v1 ve aynı taslağa uydurulmuş gürültü; ölçümler (`noise_test.json`) |
| `showcase_eski_sicaklik08.html` | İlk vitrin (sıcaklık 0,8, kısık); parametre taramaları ve kategori geçişi burada |
| `health.html` | Eğitim sağlık raporu |
| `model.pt` | Üretim ağırlıkları (EMA, fp16), 45 MB |
| `sketch_bank.pt` | Referans taslaklar; paylaşılamaz lisanslı 20 gerçek kayıt çıkarıldı |

## Kullanım

```python
from soundgen.generate import Generator
from soundgen.controls import Controls
gen = Generator('results/poc_v1/model.pt')   # sketch_bank.pt aynı klasörden okunur
out = gen.generate('gunshot', Controls(distance=0.5, environment='hall'))   # sıcaklık 1.0, son işlemsiz
```

## Etiketler (parça sayısı)

other (16000), car (5157), rain (2826), birds (2471), explosion (1988), keyboard (1896), footsteps (1749), dog_bark (1616), fire (1431), thunder (1012), gunshot (876), cough_sneeze (811), sea_waves (734), wind (723), glass_break (631), door_knock (555), helicopter (40)

## Lisanslar (parça sayısı)

CC0 1.0: 14585, CC BY 3.0: 13901, CC BY 4.0: 5545, CC BY-NC 3.0: 5383, CC SAMPLING+ 1.0: 1102

**Uyarı:** Eğitim verisinde ticari kullanıma izin vermeyen lisanslar var; bu modelin çıktıları ticari olarak kullanılamaz (yalnız araştırma / kavram kanıtı).
