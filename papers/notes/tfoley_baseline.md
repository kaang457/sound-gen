# T-FOLEY: ilk metinsiz baseline

Kaynaklar: [makale v1](https://arxiv.org/html/2401.09294v1), [resmî kod](https://github.com/YoonjinXD/T-FOLEY), [checkpoint / CC-BY-4.0](https://zenodo.org/records/10826692). İnceleme: 2026-10-07.

Sınıf koşullu waveform diffusion modeline temporal RMS bilgisi Block-FiLM ile veriliyor. Makale Section 2.2'de RMS frame 512, hop 128. Yedi DCASE Foley sınıfı var. Zamansal RMS, serbest bir event-roll veya pitch kontrolü değil.

## Yazarın bildirdiği sonuç

Makale Table 1, BFiLM: 74M parametre, 9.5 s inference, E-L1 0.0367, FAD-P 41.59, FAD-V 36.09, IS 1.79. Aynı tablodaki DAG E-L1 0.2212. Bunlar yazarın protokolüne ait; bizim CPU süremiz veya tek örnek özellikleriyle eşdeğer değil. Donanım ve tam değerlendirme protokolü eşitlenmeden hız/kalite karşılaştırması yapılmaz.

## Bizim uygulamamız

[`src/baselines/tfoley_inference.py`](../../src/baselines/tfoley_inference.py) resmî revizyon `7a0fb41c193625b7edbed86e823e4b91b55c886c` ile çalışır. Resmî model/sampler/SDE ve EMA ağırlıkları kullanılır. RFF embedding içindeki hardcoded CUDA, `sigma.device` ile düzeltilir; matematik ve ağırlıklar değiştirilmez. Ses I/O soundfile, checkpoint yükleme map_location + weights_only kullanır. Kaynak CLI'nin eski torchaudio backend bağımlılığına ihtiyaç yoktur.

Checkpoint ZIP MD5: `3eec767874a780a48e7eea2a415d1f94`. Resmî inference davranışına uygun RMS ardından elliptic smoothing, 100 sampling adımı, guidance 3 ve 20 Hz high-pass kullanılır. Dört saniye, 22050 Hz mono çıktı. Bu makalenin yeniden eğitimi veya benchmark reprodüksiyonu değildir.

## Projemiz için yorum

Çalıştırılabilir ve prompt gerektirmeyen güçlü bir başlangıç noktası. Ancak RMS dışındaki akustik parametreleri aynı modelle kontrol edemeyiz. Sonraki kontrollü deneyde seed ve sınıfı sabitleyip RMS gain ve pulse onset değiştirilecek. Sonuçlarda hedef RMS hatası ve temporal kayma kaydedilecek; kalite ayrıca dinleme ve uygun dataset ölçümleriyle değerlendirilecek.
