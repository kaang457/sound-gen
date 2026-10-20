# results

Her eğitim koşusu için tek, sade bir klasör (`02` hücre 3 eğitim bitince otomatik yazar):

```
results/<RUN>/
  README.md        özet: adım, kayıp, veri, kategoriler, nasıl kullanılır
  showcase.html    bütün sesler tek sayfada: gerçek/üretilen, parametre taramaları, kategori geçişi, varyasyonlar
  health.html      eğitim sağlık raporu
  model.pt         üretim ağırlıkları (EMA, fp16)
  sketch_bank.pt   referans taslaklar (paylaşılamaz lisanslı gerçek kayıtlar çıkarılmış)
```

`showcase.html` ve `health.html`'i indirip tarayıcıda aç. Checkpoint'ler, ara örnekler ve loglar Drive'da kalır
(`MyDrive/sound-gen/runs/<RUN>/`).

Eski çıktılar (base_v1 sesleri, son işlem denemeleri, referanslar, base_v1 sağlık raporu) **`archive`** dalında:
https://github.com/kaang457/sound-gen/tree/archive/samples
