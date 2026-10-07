# Foley adapter için veri protokolü

## Kayıt manifesti

Her satır: `recording_id`, `path`, `family`, `event_id`, `license`, `source_url`, `split`, `controls`, `control_provenance`. Aynı orijinal recording'in crop/augment versiyonları aynı split'te kalır. Train/validation/test ses ve parametre aralıkları raporlanır; kayıttan çıkarılamayan malzeme/kuvvet etiketleri uydurulmaz.

`rms_envelope` ve `centroid_hz` modelin mono16k / mean-center / peak .5 preprocessing'i **sonrası** ölçülür. Dolayısıyla ilk adapter'ın RMS kontrolü kaynağın kalibre edilmiş fiziksel şiddeti değildir. Preprocessing'in kaldırılması veya ölçek metadata'sıyla değiştirilmesi ayrı ablation gerektirir. FFT1024/hop160; RMS rectangular, centroid magnitude-weighted Hann; silence mask/probability girdisi sonraki etiket protokolünde tanımlanmalı. Quiet-frame centroid'inin anlamlı timbre hedefi sayılması değerlendirme riski olarak kaydedilir.

Fiziksel `force_n` için kalibre ölçüm; `material` için gerçek metadata veya kontrollü kayıt gerekir. `configs/foley_controls.json` içindeki örnek aralıklar doğrulanmış dataset coverage değildir. Akustik öznitelikten fiziksel kuvveti çıkarılmış gerçek etiket gibi sunmuyoruz.

## Eğitim akışı

1. Kaynak/lisans checksum manifesti ve recording bazlı split.
2. Modelin kendi mel/codec'iyle latent ve CLAP audio koşulunu offline çıkarma; cache modeller dışında Git'e eklenmez.
3. Yalnız train kayıtlarından prototype/normalization öğrenme; test sesini prototype bankasına katmama.
4. Frozen AudioLDM epsilon tahmini + residual kontrol adapter'ını gerçek latentlerle eğitme. Controls dropout ile eksik-parametre davranışı ayrıca öğretilmeli; kaynak kodda presence mask tek başına bunu kanıtlamaz.
5. Validation ile eğitim kararı, sonra ayrılmış test ve yeni kayıt/seed kontrol taraması.
6. Yeni encoder eklendiğinde eski kontrol/ses yeteneklerini aynı regression setinde değerlendirme. Parametrelerin fiziksel bağımlılıkları ayrı açıklanır.

`architecture_smoke.py` tek gerçek kayıt ve iki adımla teknik bağlantıyı denetler; bu veri/eğitim protokolünün yerine geçmez. Tam veri manifesti mevcut değildir. Şu anda yalnız tek CC0 Footstep kaynaklı prototype örneği vardır.

## Veri kaynağı adayları

[DCASE2023 Task7 resmî sayfası](https://dcase.community/challenge2023/task-foley-sound-synthesis) sınıf temelli Foley veri/evaluation referansıdır; [FSD50K resmî kayıt](https://zenodo.org/records/4060432) daha geniş olay etiketleri ve Freesound metadata içerir. Bu datasetler burada indirilip adapter eğitiminde kullanılmadı. Sınıf etiketi fiziksel kuvvet/malzeme etiketi yerine geçmez; kullanacağımız kayıtların kaynak/lisans ve metadata kapsamı manifest bazında denetlenmelidir.
