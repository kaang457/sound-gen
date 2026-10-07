# F-RAVE: sürekli çoklu kontrol için öncelikli yöntem

**Continuous descriptor-based control for deep audio synthesis**, Devis ve diğerleri, ICASSP 2023. [Makale](https://arxiv.org/html/2302.13542v1), [resmî kod](https://github.com/neurorave/neurorave/tree/main/code), [örnekler](https://neurorave.github.io/neurorave/). İnceleme: 2026-10-07.

RAVE tabanlı sentezde sürekli ses tanımlayıcıları koşul olarak kullanılır. Latent temsilin kontrol özniteliklerini taşımaması için adversarial latent confusion uygulanır. RMS, spectral centroid/bandwidth, sharpness ve booming gibi tanımlayıcılar kullanılıyor; serbest metin koşullaması yok. Deneyler alanlara özgü ses koleksiyonlarında yapılıyor; genel Foley kapsamı gösterilmiş sayılmaz.

## Kod erişimi

Kök README'deki gelecekte release ifadesine rağmen `code/` altında kaynak mevcut. `code/config/fader_config.yaml` descriptor koşullarını içeriyor. `code/src/generate.py` yazarın yerel dataset/model yollarını sabitliyor. Repodaki `VAE/trained_models` altındaki mono-attribute CVAE/VAE dosyaları, uygun çoklu kontrol F-RAVE checkpoint'i diye kabul edilmedi. Hazır Foley checkpoint bu incelemede doğrulanmadı; inference çalıştırılmadı.

## Yazar sonuçları ve sınır

Makale Table 1 (NSynth): RMS-only kontrol Spearman korelasyonu C-RAVE 0.890, F-RAVE 0.917; tüm özniteliklerde C-RAVE 0.425, F-RAVE 0.445. Table 2, ayrı centroid modeli F-RAVE 0.282 ve C-RAVE 0.310. Dolayısıyla F-RAVE her descriptor'da üstün veya çoklu kontrolde güçlü bağımsızlık garantisi veren bir çözüm değil.

Bunlar yazarın ölçümleri; bizim inference sonucumuz değil. Bizim yorumumuz: metinsiz çoklu kontrol yapısı araştırmaya uygun, fakat özellikle centroid ve çoklu kontrolün sınırlı korelasyonu somut araştırma boşluğu adayı. Dataset ve protokol farklıyken bu skorlar T-FOLEY E-L1 veya FAD değerleriyle sıralanamaz.

## Projemiz için yorum

Araştırma hedefimize T-FOLEY'den daha yakın bir çoklu kontrol yöntemi adayı: metin yerine doğrudan sayısal tanımlayıcı, latent çeşitlilik ve öznitelik ayrıştırması. Çalıştırmaya geçmeden veri alanı, descriptor normalizasyonu, uygun checkpoint ve eğitim gereksinimi netleşmeli. Mevcut RMS/centroid modülü kontrol ölçümlerine hazırlık; F-RAVE extractor'larının birebir karşılığı değil. Pitch kontrolü bu notta doğrulanmış bir F-RAVE özelliği olarak sunulmuyor.
