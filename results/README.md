# results

Her eğitim koşusu için tek, sade bir klasör (notebook eğitim bitince otomatik yazar):

```
results/<vocoder RUN>/     02_vocoder
  README.md        özet: adım, kayıp, veri, mel sözleşmesi
  showcase.html    gerçek kayıt ile aynı mel'den vocoder çıktısı yan yana, parametre taramaları
  health.html      eğitim sağlık raporu
  model.pt         vocoder ağırlıkları (EMA, fp16)
  sketch_bank.pt   vitrin için örnek mel'ler

results/<mel üretici RUN>/ 03_mel_generator
  index.html       sıfırdan üretilen sesler: mel resmi, ses, en yakın eğitim kaydına uzaklık, CLAP tahmini
  *_sesler.zip     bütün sesler
  scratch.json, clap.json, loss.csv, mel_gen.pt (fp16)
```

`poc_v1/` eski yolun (referans taslak + kategori koşullu WaveFlow, 22 kHz) sonucudur.

`showcase.html` ve `health.html`'i indirip tarayıcıda aç. Checkpoint'ler, ara örnekler ve loglar Drive'da kalır
(`MyDrive/sound-gen/runs/<RUN>/`).

Eski çıktılar (base_v1 sesleri, son işlem denemeleri, referanslar, base_v1 sağlık raporu) **`archive`** dalında:
https://github.com/kaang457/sound-gen/tree/archive/samples
