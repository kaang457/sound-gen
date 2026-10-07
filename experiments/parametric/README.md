# Sayısal kontrollü neural ses üretimi

**Rol:** Bu klasör önceki sınırlı kontrol deneylerini saklar. Küçük spectral CVAE ana model değildir; iki-parametre başarısı güncel genişletilebilir Foley hedefini karşılamaz. Aktif kapsam [PROJECT_SCOPE.md](../../docs/PROJECT_SCOPE.md), sıradaki çalışma [ana mimari seçimi](../../docs/ROADMAP.md).


09: pretrained TFOLEY'ye RMS vektörünü doğrudan verme. 10: kendi eğitilmiş küçük spectral CVAE'mize RMS ve centroid verme. İkisinde de kullanıcı text prompt yok. `src/control/generate.py` ortak giriş noktasıdır; desteklenmeyen kontrol hata verir.

```python
from pathlib import Path
from src.control.generate import generate
ROOT = Path.cwd()  # sound-gen proje kökü
waveform, sr, metrics = generate('spectral_cvae', ROOT,
    rms=0.025, centroid_hz=1800, duration_s=2, seed=42, name='my_sound')
```

CVAE eğitimli `spectral_cvae/weights.json` ile gelir; ilk denemek için yeniden eğitim veya model indirmesi gerekmez. Mevcut PyTorch/TFOLEY ortamını kullanın. `pip install -r requirements.txt -r requirements-tfoley.txt` yardımcı paketleri kurar. WAV'lar yerelde üretilir, Git'te kod, küçük JSON ağırlıklar ve ölçüm sonuçları bulunur.

Proje kökünde yeniden üretim:

```powershell
python src/control/tfoley_sweep.py --prepare --steps 50 --device auto
python src/control/spectral_cvae.py --train --steps 3000
python -m unittest discover -s tests
```

TFOLEY: 3 eğri × 2 seed, 50 adım, Footstep, guidance 3. Gaussian hedef peak RMS .015/.045 ve zaman kaydırma .03; hedef vektör aynen koşuldur, sonradan gain eşleştirmesi yok. RMS ölçümü 512/128, ellip(4,.01,120,.125) gust smoothing. Peak tanısı 250 ms mesafe/%15 prominence; Hungarian bire bir eşleştirme, 150 ms kabul. Olay sınıfı/algısal kalite ölçülmez.

CVAE: 4 s CC0 freefire66 kaydı, 512 örneklik nonoverlap rectangular FFT blokları; 103 train, 34 validation, 35 geliştirme tanı bloğu. Eğitim seed17, 3000 Adam adımı, batch64, lr .002. RMS/centroid model decoder'a koşul olarak girer. Pozitif magnitude çıktısı inverse FFT + rastgele fazla render edilir; çıkış genliği/centroid'i hedefe zorlayan DSP uygulanmaz. Random-phase rendering zamansal yapı ve doğal ayak basma niteliğini korumaz. Süre tam bloklara yuvarlanır.

İkinci eğitimde son zaman bölümü prior sampling tanısı: RMS relative MAE **%4.47**, centroid MAE **30.82 Hz**. Bu aynı kayıt üzerindeki geliştirme tanısıdır, bağımsız benchmark değildir. Üç seed'in tamamında iki kontrolün taraması monoton; RMS değişirken centroid drift 37–39 Hz, centroid değişirken RMS drift yaklaşık %1.8–2.2. Bunlar tanımlı küçük kontrol aralığındaki sonuçlardır. `spectral_cvae/sweep.json` 12 klibin ayrıntısını, `training.json` eğitim tanılarını, `initial_trial.json` başarısız ilk denemeyi saklar. Ortak-gain dinleme notebook10 içinde yapılır; burada algısal kalite sonucu yoktur.

[Model seçimi / makale eşleşmesi](../../papers/notes/parametric_model_selection.md). Pitch kontrolü ve genel Foley generation henüz bu prototipin yetenekleri değildir.

## TFOLEY tamamlanan tarama

İki seed için de yüksek hedef RMS, düşük hedefe göre daha yüksek çıktı RMS üretti. Bire bir peak tanısı ve bütün hatalar `tfoley_sweep.json` / `results/tables/parametric/tfoley_sweep.csv` içinde. Bu 6 örnekli CPU taraması ses kalitesi veya genel başarı benchmark’ı değildir.
