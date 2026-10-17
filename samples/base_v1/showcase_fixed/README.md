# base_v1 (adım 11.783) — son işlem denemesi

Aynı model çıktıları (`../showcase/*_gen.wav`), yeniden eğitim yapılmadan iki son işlemle:
1. **1378 Hz tarak filtresi** — 22050/16 Hz ve katlarındaki sabit tonlar (WaveFlow'un h=16 katlamasından gelen vızıltı)
2. **Taslağa zarf eşleme** — üretilen sesin 20 bantlık kaba spektrumu, koşul taslağına zorlanır

Dosyalar, her kategori için:
- `<kategori>_fixed.wav` — model + son işlem
- `<kategori>_noise_baseline.wav` — **kontrol**: beyaz gürültü, aynı taslakla aynı şekilde şekillendirilmiş (model yok)
- gerçek ses: `../showcase/<kategori>_real.wav`

Model, gürültü kontrolünden belirgin şekilde daha gerçekçi değilse ince ayrıntıyı henüz öğrenmemiş demektir.
