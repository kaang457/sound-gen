# poc_v1_dry

- Adım: **1298** · en iyi doğrulama NLL: **-2.0018** (düşük = iyi)
- Veri: 47.8 saat, 40516 parça, 17 etiket
- Model: 8 flow × 8 katman, 128 kanal, upsampler `interp`, cond_norm True, actnorm True
- Vitrin son işlemi: `refine='v1'`

| Dosya | Ne |
|---|---|
| `showcase.html` | Bütün sesler tek sayfada (indirip tarayıcıda aç) |
| `health.html` | Eğitim sağlık raporu |
| `model.pt` | Üretim ağırlıkları (EMA, fp16), 45 MB |
| `sketch_bank.pt` | Referans taslaklar; paylaşılamaz lisanslı 20 gerçek kayıt çıkarıldı |

## Kullanım

```python
from soundgen.generate import Generator
from soundgen.controls import Controls
gen = Generator('results/poc_v1_dry/model.pt')   # sketch_bank.pt aynı klasörden okunur
out = gen.generate('gunshot', Controls(distance=0.5, environment='hall'), refine='v1')
```

## Etiketler (parça sayısı)

other (16000), car (5157), rain (2826), birds (2471), explosion (1988), keyboard (1896), footsteps (1749), dog_bark (1616), fire (1431), thunder (1012), gunshot (876), cough_sneeze (811), sea_waves (734), wind (723), glass_break (631), door_knock (555), helicopter (40)

## Lisanslar (parça sayısı)

CC0 1.0: 14585, CC BY 3.0: 13901, CC BY 4.0: 5545, CC BY-NC 3.0: 5383, CC SAMPLING+ 1.0: 1102

**Uyarı:** Eğitim verisinde ticari kullanıma izin vermeyen lisanslar var; bu modelin çıktıları ticari olarak kullanılamaz (yalnız araştırma / kavram kanıtı).
