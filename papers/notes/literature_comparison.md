# Çalışma seçimi: metinsiz, ölçülebilir kontrol

> Güncel yön: [parametrik Foley kapsamı](../../docs/PROJECT_SCOPE.md) ve [ilerleme planı](../../docs/ROADMAP.md). Bu not önceki deneyleri/kod denetimini belgeliyor; TFOLEY, AudioLDM, küçük CVAE veya F-RAVE ana üretim modeli olarak seçilmiş değildir. Seçim, Foley kalitesi ve yeni parametre ekleme kapasitesi üzerinden yapılacak.


Kontrol tarihi: 2026-10-07. Bu karar kod ve kaynak incelemesine dayanır; bütün adayların aynı veri üzerinde karşılaştırıldığı bir benchmark değildir.

## Referans sesi taklit etme hedefi için güncelleme

İlk dinlemede beklenen benzerlik sağlanmadı. T-FOLEY yalnız temporal baseline olarak tutuluyor. Kuş/çevresel referansın içeriğini koşul olarak kullanmak için **AudioLDM audio-to-audio** yeni öncelikli inference adayı; caption gerektirmeyen yol kodda doğrulandı, ilk kuş referansı koşumu çalıştırıldı; algısal başarı ölçülmedi. Sayısal çoklu kontrol hedefinde F-RAVE yöntem adayı sürüyor. [Kod denetimi ve güncel seçim](reference_audio_selection.md).

## Önceki temporal / descriptor seçimi

**İlk çalıştırılabilir baseline: T-FOLEY. F-RAVE çoklu öznitelik koşullandırması için yöntem referansıdır; ana mimari seçilmedi.**

Projeye uygunluk ölçütümüz, serbest metin kullanmadan ses sınıfı ve sayısal kontrol sunabilmek. Hazır checkpoint erişimi ikinci ölçüt; pitch/parlaklık gibi birden fazla özelliği bağımsız değiştirmek üçüncü ölçüt. Bir modelin boş caption kabul etmesi, metinsiz koşullamada eğitildiğini veya sınıf anlamını koruduğunu kanıtlamaz.

T-FOLEY sınıf + zamana bağlı RMS ile bu ölçütlerin ilk ikisini karşılıyor. F-RAVE sürekli tanımlayıcılar ve latent öznitelik ayrıştırması nedeniyle araştırma sorumuza daha yakın; geniş Foley kapsamı ve doğrudan çalıştırılabilir uygun checkpoint doğrulanmadığı için ilk inference olarak seçilmedi. **T-FOLEY'nin başarılı çalışması bütün parametre uzayını çözdüğümüz anlamına gelmez.**

## Karşılaştırma

| Çalışma | Belgelenmiş koşullama / metin ilişkisi | Kontroller | Kod / uygun hazır ağırlık | Bu projedeki karar |
|---|---|---|---|---|
| [T-FOLEY](https://arxiv.org/html/2401.09294v1) | Sınıf etiketi + RMS eğrisi; text encoder gerektirmez | Enerji zarfı ve dolaylı olay zamanlaması; 7 sınıf | [Resmî kod](https://github.com/YoonjinXD/T-FOLEY), [checkpoint](https://zenodo.org/records/10826692) indirildi ve checksum doğrulandı | İlk temporal-control baseline; pitch ve parlaklık kontrolü yok |
| [F-RAVE](https://arxiv.org/html/2302.13542v1) | Latent + sürekli akustik tanımlayıcılar; serbest metin yok | RMS, centroid, bandwidth, sharpness, booming; konfigürasyona bağlı | [Kod](https://github.com/neurorave/neurorave/tree/main/code) mevcut; genel Foley için uygun F-RAVE checkpoint doğrulanmadı | Sürekli çoklu kontrol ve disentanglement için ilk yöntem adayı |
| [Sketch2Sound](https://arxiv.org/html/2412.08550v2) | Text-to-audio backbone + kontrol; makalede text dropout var, metni kaldırmak teknik olarak mümkün | Loudness, brightness, pitch, sonic imitation | Bu incelemede çalıştırılabilir resmî checkpoint doğrulanmadı | Descriptor ve mimari referansı; metinsiz semantik başarı ayrıca gösterilmeli |
| [Audio ControlNet](https://github.com/juhayna-zh/AudioControlNet) | Belgelenen inference caption + kontrol kullanıyor; caption-free başarı doğrulanmadı | Loudness, pitch, sound-event roll | Kod ve Hugging Face adapter ağırlıkları var | Hibrit karşılaştırıcı; ana metinsiz baseline değil |
| [Multiple Conditional Diffusion](https://ojs.aaai.org/index.php/AAAI/article/view/29773) | Metin üzerine ek koşullar | Timestamp, pitch, energy | Kullanılabilir resmî inference/checkpoint bu incelemede doğrulanmadı | Çoklu koşul birleştirme referansı |
| [PicoAudio](https://github.com/zeyuxie29/PicoAudio) | Text-to-audio; olay ve zaman yapısı; LLM önişleme ayrılabilir | Olay sayısı ve zamanlaması | Resmî kod mevcut; bu ortamda inference yapılmadı | Temporal referans; LLM'yi kaldırmak text encoder'ı kaldırmaz |
| [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) | Yapılandırılmış metin; manuel yapı ChatGPT önişlemesini atlayabilir | Olay ve zaman ilişkileri | Resmî kod / model indirme yönergeleri mevcut; denenmedi | Prompt koşullama sürüyor; düşük öncelik |
| [Stable Audio Open](https://arxiv.org/abs/2407.14358) | Text encoder + latent diffusion | Caption ve süre | [Inference kütüphanesi](https://github.com/Stability-AI/stable-audio-tools); demo repo yalnızca demo kaynağı | Metin baseline/backbone adayı; ana kontrol çözümümüz değil |

"Doğrulanmadı" yayımlanmadığı anlamına gelmez. Burada doğrulanmış, denenmiş ve yalnız belgelenmiş erişim ayrı tutuluyor. PicoAudio'daki event frequency olay tekrar sayısıdır; akustik frekans (Hz) veya F0 değildir.

## Araştırma kararı ve sınırları

Bizim yorumumuz: RMS ve temporal kontrolle bir ilk ölçüm protokolü kurup, F-RAVE'nin descriptor conditioning ve öznitelik ayrıştırma yaklaşımını çoklu kontrol deneyi için önceliklendirmek en uygun yol. Hazır text-to-audio modellere adapter eklemek, metin bağımlılığını kendiliğinden gidermez.

T-FOLEY yalnız Footstep gibi bir sınıf ID'sini değiştirerek semantik kontrol sağlıyor; malzeme veya geniş açık sınıf sözlüğü sunmuyor. RMS fiziksel genlik/enerji proxy'si; perceptual loudness değildir. Parlaklık için centroid kullanmak, algısal doğrulama gereksinimini ortadan kaldırmaz. F0 yalnız güvenilir tonal karelerde değerlendirilmelidir; gürültülü Foley'de her kareye pitch atamak hatalı olabilir.

Bir sonraki seçim eşiği: T-FOLEY'de sabit seed/sınıf ile RMS/onset sweep; F-RAVE için veri alanına uygun checkpoint ya da yeniden eğitim maliyetinin denetlenmesi. Çoklu kontrol iddiası için hedef hata, monotonluk ve diğer özniteliklerdeki değişim birlikte ölçülecek. İlk tek örnek inference ses kalitesi veya model üstünlüğü sonucu değildir.

## Kaynak ve uygulama izi

- [T-FOLEY ayrıntılı notu](tfoley_baseline.md), [F-RAVE ayrıntılı notu](frave_controls.md)
- [Makine okunur seçim tablosu](../../results/tables/literature_comparison.csv)
- [Kod–literatür eşleşmesi](code_literature_map.md)
- [Baseline koşum kaydı](../../experiments/baselines/tfoley/)

Yazar sonuçları kendi deney sonuçlarımızdan ayrı tutulur. FAD değerleri farklı embedding, veri, örnek sayısı ve protokol üzerinden doğrudan sıralanmaz.


## Parametrik prototip güncellemesi

09 doğrudan TFOLEY RMS taraması, 10 ise kendi küçük spectral CVAE'mizin RMS/centroid kontrolünü içerir. CVAE tek gerçek kayıtla eğitildi; F-RAVE gerçeklenmesi veya doğal Foley üstünlüğü sonucu değildir. F-RAVE yöntem adayı, TFOLEY temporal baseline, AudioLDM referans-içerik baseline olarak kalır. [Ölçülen sonuçlar ve seçim](parametric_model_selection.md).
