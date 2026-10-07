# Parametric Foley Generation

Bu projenin hedefi, **çoğunlukla Foley sesleri üreten ve üretimi genişletilebilir sayısal parametrelerle yönlendirilebilen bir deep learning sistemi** geliştirmektir. Kullanıcı ses üretmek için doğal dil prompt'u yazmak zorunda olmamalıdır. Yeni kontrol parametreleri eklemek, ana araştırma hedefinin bir parçasıdır.

Başarı, yalnız RMS/centroid hatasının düşük olması değildir: sesin kullanılabilir kalitede olması, istenen Foley olayına uyması ve parametre değişikliğinin duyulabilir, ölçülebilir karşılık üretmesi birlikte gerekir.

## Güncel kapsam

- Foley üretimi: adım, darbe, sürtünme, nesne hareketi gibi olay aileleri. İlk aile doğrulama alanıdır; sistemin nihai kapsamını tek sınıfa kapatmaz.
- Olay/fiziksel kontroller: malzeme, yüzey, kuvvet, boyut, hareket hızı gibi ses türüne göre anlamlı parametreler.
- Akustik kontroller: şiddet, parlaklık, rezonans, sönüm; pitch yalnız anlamlı olduğu seslerde.
- Zaman kontrolleri: başlangıç, süre, tekrar sıklığı ve zamanla değişen eğriler.
- Yeni parametreler ve eksik/uygulanamaz kontroller için açık bir koşullandırma tasarımı.

Bunlar **hedef yeteneklerdir**; mevcut modellerin bunları desteklediği iddia edilmiyor. Parametrelerin öğrenilebilir olması için uygun veri/etiket ve eğitim gerekir. Video-to-audio, genel müzik üretimi ve yalnız referans sesi taklit etmek ana kapsam değildir. Referans ses isteğe bağlı bir koşul olabilir.

[Proje kapsamı ve başarı ölçütleri](docs/PROJECT_SCOPE.md) · [İlerleme planı](docs/ROADMAP.md) · [Mimari seçim ölçütleri](papers/notes/parametric_model_selection.md)

## Mevcut durum ve deneylerin rolü

**Ana üretim modeli henüz seçilmedi.** Hazır kaliteli bir modelin sayısal koşullandırmaya uyarlanması ile kendi modelimizi eğitme seçenekleri, Foley kapsamı, kalite, genişletilebilirlik ve eğitim maliyeti üzerinden karşılaştırılacak. İç mimaride text encoder bulunması tek başına eleme nedeni değildir; kullanıcının prompt'a ihtiyaç duymadan hedef sesi ve kontrolleri verebilmesi gerekir.

| Çalışma | Güncel rol | Henüz göstermediği |
|---|---|---|
| TFOLEY, 06–07–09 | Pretrained sınıf + RMS/zaman baseline | Geniş Foley kapsamı, çoklu fiziksel/akustik kontrol, doğrulanmış kullanılabilir kalite |
| AudioLDM, 08 | Referans-audio koşullandırma deneyi | Bağımsız, genişletilebilir sayısal kontrol |
| Kendi spectral CVAE'miz, 10 | Küçük iki-kontrol mühendislik deneyi; ana model değil | Doğal Foley, temporal yapı ve kayıtlar arası genelleme |
| F-RAVE | Descriptor conditioning / ayrıştırma için mimari referans | Bu projeye uygun hazır Foley modeli veya burada doğrulanmış kalite |

09'daki altı TFOLEY üretimi enerji kontrolüne, 10'daki tek-kayıt CVAE deneyi iki sayısal koşulun öğrenilmesine dair sınırlı kanıt sağlar. Bu sonuçlar ana hedefin tamamlandığı anlamına gelmez. [Deney sonuçları](experiments/parametric/README.md).

## Sıradaki teslim

Yeni bir küçük sentez demosu yerine, **ana mimari için gerekçeli seçim**: Foley üretebilen adayların dinlenebilir örnekleri, parametre koşullandırma yolu, yeni parametre ekleme tasarımı, veri/etiket ihtiyacı ve donanım maliyeti birlikte değerlendirilecek. Model seçimi öncesinde kontrol eklemenin temel ses kalitesini koruyabileceği gösterilmeli.

## Notebooklar

| Notebook | Amaç |
|---|---|
| [01 Audio basics](notebooks/01_audio_basics.ipynb) | Sinyal analizi hazırlığı |
| [02 Feature extraction](notebooks/02_feature_extraction.ipynb) | Öznitelik ölçümü hazırlığı |
| [03 Brightness at constant RMS](notebooks/03_brightness_at_constant_rms.ipynb) | Ölçüm/davranış deneyi |
| [04 Real audio analysis](notebooks/04_real_audio_analysis.ipynb) | Lisanslı gerçek kayıt analizi |
| [05 Pitch / periodicity](notebooks/05_pitch_periodicity.ipynb) | Pitch ölçümünün uygulanabilirliği |
| [06 TFOLEY inference](notebooks/06_tfoley_inference.ipynb) | Hazır-model inference deneyi |
| [07 Real Footstep](notebooks/07_tfoley_real_footstep.ipynb) | Gerçek referansın RMS/zaman tanısı |
| [08 AudioLDM reference](notebooks/08_audioldm_reference_audio.ipynb) | Referans içerik koşullandırma deneyi |
| [09 Numeric TFOLEY](notebooks/09_parametric_tfoley.ipynb) | Doğrudan RMS kontrol baseline'ı |
| [10 Spectral CVAE](notebooks/10_parametric_spectral_cvae.ipynb) | Sınırlı iki-kontrol prototipi |

Notebook numaraları tarihsel deney sırasıdır, ana modelin geliştirme aşamaları değildir. [Kod–makale eşleşmesi](papers/notes/code_literature_map.md) uygulanan bileşenleri ve sınırları kaydeder.

## Dosyalar ve yeniden üretim

`docs/`: aktif kapsam/plan. `papers/notes/`: model ve çalışma incelemeleri. `src/`: çalıştırılabilir kod. `experiments/`: koşum, kaynak ve sonuç kayıtları. `results/`: ölçüm tabloları/grafikleri. Önceki README [tarihsel kayıtta](docs/history/README_before_scope_revision.md) korunur; güncel yön bu sayfa ve kapsam belgesidir.

Temel ortam Python 3.11: `pip install -r requirements.txt`. TFOLEY/CVAE için mevcut PyTorch kurulumuna ek `pip install -r requirements-tfoley.txt`. AudioLDM ayrı ortam kullanır; kurulum 08 notebookunda. Büyük dış model ağırlıkları ve WAV'lar yerelde tutulur; küçük CVAE JSON ağırlıkları mühendislik deneyini yeniden çalıştırmak için repodadır. Deneyleri yeniden çalıştırmak, ana mimari seçiminden önce zorunlu bir sonraki adım değildir.
