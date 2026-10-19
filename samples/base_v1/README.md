# base_v1

İlk eğitilmiş model: 15 kategori, 11.783 adım, 1,5 A100-saat (DCASE + ESC-50; ticari kullanıma uygun değil).

| Klasör | Ne | Durum |
|---|---|---|
| `showcase/` | Modelin ham çıktısı (`best.pt`): gerçek ve üretilen, kategori geçişi, parametre taramaları | **Değerlendirilecek olan bu** |
| `step_011783/` | Eğitimin son adımında yazılan gerçek/üretilen çiftleri | Ham çıktı, eğitim sırasında |
| `health/report.html` | Eğitim sağlık raporu | Bkz. `docs/PROCESS.md` §9a |
| `rejected/postprocess_v1/` | Ham çıktıya uygulanan 1. son işlem denemesi (notch filtre + zarf eşleme) | ❌ Reddedildi: cızırtı farklı tonda sürdü |
| `rejected/postprocess_v2/` | 2. son işlem denemesi ("v2") | ❌ Reddedildi ve kodu geri alındı: ya gürültü ya sessizlik |

`rejected/` altındakiler aynı modelin çıktısına sonradan uygulanmış işlemler; ayrı modeller değil. Ayrıntılar: `docs/PROCESS.md` §6–7.
