# results

Her eğitim koşusu için `02_train` hücre 4'ün yazdığı klasör:

```
results/<RUN>/
  index.html        istek başına iki ses (varyasyon A ve A→B karışımı): mel resmi, oynatıcı, en yakın eğitim kaydına
                    uzaklık, olay sayısı, CLAP istek eşleşmesi
  <RUN>_sesler.zip  bütün istekler × 8 aday
  scratch.json, clap.json, pairs.json, loss.csv
```

Model (`ema.pt`, ~0,6 GB) GitHub sınırını aştığı için Drive'da (`MyDrive/sound-gen/runs/<RUN>/`) kalır.
Araştırma dönemi sonuçları (poc_v1 vb.): `research-v1` dalı.
