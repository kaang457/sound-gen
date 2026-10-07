# Ana mimari için ilk gerçek entegrasyon

Bu çalışma önceki küçük CVAE'den ayrıdır. AudioLDM m-full'ün gerçek pretrained backbone'u üstüne genişletilebilir control bank + residual epsilon adapter bağlandı. [Karar ve alternatifler](../../docs/decisions/001_foley_conditioning_architecture.md).

## Çalıştırılanlar

- Frozen pretrained model üzerinde gerçek Footstep mel/codec latentleriyle iki optimizer adımı: `audioldm_adapter_smoke.json`. Latent `[1,8,128,16]`, CLAP koşulu `[1,1,512]`; adapter 41,288 parametre. Gradientler sonlu ve sıfırdan farklı; backbone gradienti yok.
- Sıfır başlangıç residual'ı epsilon tahminini değiştirmiyor; hiçbir kontrol yoksa optimizer sonrası da sıfır katkı veriyor.
- Kullanıcı prompt'u ve runtime referans WAV olmadan, offline hazırlanan **tek örnekli** Footstep audio prototype ile 50-adım/5-s baseline üretildi. Bu kategori adıyla text conditioning değildir. `footstep_prototype_baseline.json`, ayrıca structured CLI koşumu `structured_footstep_baseline_run.json` sonucu kaydeder.
- Gerçek tek-kayıt latent cache hazırlığı tamamlandı; bu kayıt production eğitim guard’ını geçemez.
- Typed scalar/curve/categorical registry, geçerlilik, eksik/sıfır ayrımı, yeni eksik encoder eklenince eski alanın/residual'ın korunması ve recording leakage guard testleri.

**İki adım, kontrol öğrenilmiş model sonucu değildir.** Baseline dinlenebilir fakat kalitesi burada değerlendirilmedi. Fiziksel kontrol etiketli veri ve çok-kayıt adapter eğitimi tamamlanmadı. Büyük eğitim çalıştırılmış veya bütün Foley kapsamı çözülmüş gibi raporlamıyoruz.

## Kullanım

08 notebookundaki `.venv-audioldm` ortamı ve checkpoint/source hazırlığı kullanılır. Önce 08 kurulumunu tamamlayın; `--prepare` ile büyük dosyalar orada edinilir. Windows proje kökünde:

```powershell
.\.venv-audioldm\Scripts\python.exe src/foley/infer.py --request configs/foley_baseline_request.json --device auto --output-name local_foley_baseline
```

`results/audio/foley/local_foley_baseline.wav` baseline çıktıdır. [11 notebook](../../notebooks/11_foley_backbone.ipynb) üretip dinletir. Sayısal kontrol isteği `foley_target_request.json` içinde örneklenir; gerçek trained adapter verilmeden açık hata üretir. Parametreyi sessizce yok sayarak baseline'ı kontrollü ses diye sunmaz.

## Hazırlanan eğitim hattı

```powershell
.\.venv-audioldm\Scripts\python.exe src/foley/prepare_cache.py --manifest YOUR_MANIFEST.json --device cuda
.\.venv-audioldm\Scripts\python.exe src/foley/build_prototypes.py --index models/foley/cache/index.json
.\.venv-audioldm\Scripts\python.exe src/foley/train_adapter.py --index models/foley/cache/index.json --device cuda --steps 1000
```

`prepare_cache.py` şu anda `cpu` veya `cuda` alır; bu komutta Windows GPU için `--device cuda`, CPU için `--device cpu` kullanın. `YOUR_MANIFEST.json` henüz sağlanmış çok-kayıt dataset değildir. [Veri protokolü](../../docs/FOLEY_DATA_PROTOCOL.md) ve `configs/foley_single_record_smoke.json` formatı gösterir; tek kayıt production eğitimine geçemez. Train scriptinin tamamlanması kalite/control accuracy değerlendirmesi değildir; validation/test inference henüz uygulanmış sonuç değildir.

Ağırlıklar ve latent cache `models/foley/` altında yereldir; repo şema, kod, kaynak provenance ve koşum kayıtlarını taşır. `extend_adapter.py` encoder eklerken eski ağırlıkları korur; yeni parametre trained-controls listesine kendiliğinden girmez. Bunun model kalitesine etkisi yeni eğitim sonrası ayrıca ölçülmelidir.

## Uyumluluk

Gerçek kontrol Linux CPU / torch2.11 ortamında yapıldı; Windows/GPU bu yeni hat için çalıştırılmadı. Legacy STFT'nin positional librosa çağrıları keyword-only API'ye uyarlanır. Pretrained inference EMA ağırlıkları UNet dondurulmadan önce yüklenir; legacy `ema_scope` frozen parametrelerde assertion yaptığı için yeni yolda scope tekrar kullanılmaz. Kaynak pin/checksum korunur.

Bellek denetimi: iki full-model subprocess eşzamanlı çalışınca Linux işleri exit137 ile sonlandı; büyük model hazırlama/inference işleri sırayla çalıştırılır. Cache hazırlığı ve yapılandırılmış inference seri yeniden doğrulandı. Bu gözlem GPU bellek maliyetinin ölçümü değildir.

Yeni-parametre modu: `freeze_for_extension(adapter, new_controls)` yalnız yeni encoder’ları eğitime açar; trunk/semantic projection/eski encoder’lar sabit tutulur. Yeni kontrol yokken eski residual’ın değişmediği bir optimizer adımıyla test edildi. Bu, yeni kontrolün başarılı veya başka seslere genellenebilir öğrenildiğini kanıtlamaz. CLI eğitim örneği başlangıçtan adapter eğitimi içindir; extension modu helper API üzerinden uygulanır.
