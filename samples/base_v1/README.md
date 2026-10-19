# base_v1

İlk eğitilmiş model: 15 kategori, 11.783 adım, 1,5 A100-saat (DCASE + ESC-50; ticari kullanıma uygun değil).

| Klasör | Ne | Durum |
|---|---|---|
| `showcase/` | Modelin çıktısı (`best.pt`) + standart hafif temizlik (alçak frekans süzgeci, sabit ton bastırma, kenar yumuşatma, limiter): gerçek ve üretilen, kategori geçişi, parametre taramaları | **Değerlendirilecek olan bu** |
| `step_011783/` | Eğitimin son adımında yazılan gerçek/üretilen çiftleri | Ham çıktı, eğitim sırasında |
| `health/report.html` | Eğitim sağlık raporu | Bkz. `docs/PROCESS.md` §9a |
| `postprocess_v1/` | Son işlem v1: 1378 Hz notch + taslağa eşleme (`*_fixed.wav`); kontrol olarak aynı işlemin beyaz gürültüye uygulanmışı (`*_noise_baseline.wav`) | ✅ Dinleme testinde beğenildi; üretimde `refine="v1"` (02 hücre 7 ve 04'te varsayılan) |
| `postprocess_v1/sweeps/` | `showcase/sweep_*` dosyalarına aynı v1 işlemi (her taramanın kendi düzenlenmiş taslağına göre) | ✅ |
| `rejected/postprocess_v2/` | 2. son işlem denemesi ("v2"): gürültüsü agresif bastırılmış vitrin, parametre taramaları dahil | ❌ Reddedildi ve kodu geri alındı: ya gürültü ya sessizlik |

`postprocess_v1/` ve `rejected/` altındakiler aynı modelin çıktısına sonradan uygulanmış işlemler; ayrı modeller değil. Ayrıntılar: `docs/PROCESS.md` §6–7.
