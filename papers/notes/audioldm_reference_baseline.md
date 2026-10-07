# AudioLDM: referans-ses koşullama uygulaması

Kaynaklar: [ICML 2023 makalesi](https://arxiv.org/abs/2301.12503), [resmî kod](https://github.com/haoheliu/AudioLDM), [m-full checkpoint](https://zenodo.org/records/7813012). İnceleme/koşum: 2026-10-07.

## Paper / resmî kod ilişkisi

AudioLDM, CLAP temsilinden koşullanan latent audio diffusion kullanır. Resmî repository audio-to-audio modunda referans waveform'un audio embedding'ini kullanır. Kullanıcı text prompt'u vermeden çalıştırılabilir; modelde text branch/empty unconditional embedding var olmaya devam eder. Bu bağımsız akustik öznitelik kontrolü veya birebir timbre taklidi garantisi değildir.

İncelenen resmî revizyon `4054cb418ba3d947d94f6ad302f1d6a320e2313c`: `pipeline.py::set_cond_audio`, condition key waveform/embed mode audio yapar. `ldm.py::generate_sample` birden çok adayda text similarity ile ranking uygular. `clap/encoders.py` condition dropout'unu eval'da da uyguluyor.

## Bizim uygulama

[`audioldm_reference_inference.py`](../../src/baselines/audioldm_reference_inference.py) model, CLAP, latent encoder/decoder, DDIM ve vocoder'ı resmî kaynak/EMA checkpoint üzerinden kullanır. Pozitif mode waveform/audio, koşul dropout 0, candidate 1 ve text-scoring guard. Soundfile/librosa I/O resmî mono/16k/mean-center/peak .5/crop-pad önişlemesine uyarlanır.

Checkpoint `weights_only=True` + mmap ile CPU'ya okunur. Missing/unexpected weight keys denetlenir; yalnız eski `position_ids` buffer farkına izin verilir. Kaynak kodun ayrıca indirmeye çalıştığı geçici RoBERTa encoder'ı yerel config'ten kurulur ve **bütün text ağırlıkları AudioLDM checkpoint'inden yüklenir**. Bu genel LAION CLAP encoder'ını uyumluluk varsayımıyla takmak değildir. Tokenizer/config revision ve SHA-256 pinlidir.

Inference ayrı ortam: Torch/Audio 2.11.0, Vision 0.26.0, numpy 1.26.4, librosa 0.11.0, transformers 4.44.2. Mevcut proje `.venv` değiştirilmez. Referans koşullama dropout'u ve text ranking düzeltmeleri kayıtta açıklanır; paper reprodüksiyonu diye raporlanmaz.

## Sonuç ve uygunluk

[İlk gerçek koşum](../../experiments/baselines/audioldm/README.md): kuş referansı, 50 adım, seed 42, 5 s çıktı, CPU. Üretim ve metadata doğrulandı; algısal başarı veya kategori doğruluğu ölçülmedi. Düşük seviyeli kısa etkin bölüm için ayrı normalize playback sağlanıyor; bu bir quality fix değildir.

Bizim yorumumuz: T-FOLEY'nin yedi sınıf/RMS sınırını aşarak referans içeriği kullanma hedefi için daha uygun bir **deney adayı**. Bağımsız numeric multi-control hedefini çözmediği için bütün proje için kesin seçim değil. F-RAVE descriptor yöntemi incelemesi ve uygun Foley checkpoint araştırması sürmeli. Önce dinleme/kategori uygunluğu, ardından sabit referans/çoklu seed ile tutarlılık incelenecek. Yazarın sayısal sonuçları bu tek koşumdan yeniden üretilmedi.

`src/audio/audioldm_diagnostics.py` notebook ve kaydedilen CSV/grafikler için ortak tanı kodudur. RMS/centroid ölçümleri semantic similarity skorları değildir.
