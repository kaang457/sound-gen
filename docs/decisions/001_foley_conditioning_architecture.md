# ADR 001: Ses-semantiği + modüler sayısal kontrol adapter'ı

Tarih: 2026-10-07. **Durum: ilk entegrasyon mimarisi seçildi; üretim kalitesi / kontrol başarısı için ana model kabulü bekliyor.** Bu karar küçük CVAE'yi büyütmek değildir.

## Karar

İlk mühendislik backbone'u mevcut ve gerçek inference'ı doğrulanmış **AudioLDM m-full**. Pozitif semantik koşul CLAP audio embedding'i; kullanıcı prompt'u yok. Bir Foley olayını seçmek için gerçek kayıt örneklerinden offline hazırlanmış audio prototype bankası kullanılacak. İlk bankada yalnız bir Footstep örneği var; bu geniş Foley sınıflandırıcısı veya kayıtlar arası genelleme değildir.

Frozen backbone'un epsilon tahminine **öğrenilebilir residual adapter** eklenir. Adapter noisy latent, diffusion time, CLAP audio-semantic koşulu ve her kontrol için ayrı encoder'dan gelen zamansal alanı alır. Encoder bankası sabit iki parametreye kapanmaz: scalar/curve/categorical parametreler bir registry üzerinden eklenir. Eksik parametre alanı sıfır katkı verir; gerçek sıfır değerle aynı değildir. Son katmanın sıfır başlatılması başlangıçta pretrained epsilon tahminini korur. Kayıp epsilon MSE'dir; gerçek Foley verisiyle adapter eğitimi ve kalite/kontrol değerlendirmesi gerekir.

Bu residual tasarım bizim ilk prototipimizdir; Audio ControlNet veya Sketch2Sound mimarisinin yeniden üretimi değildir. Expressivity/kalite yetersiz kalırsa UNet içi çok-ölçekli conditioning veya DiT backbone'u değerlendirilecek. Bunu söylemek mevcut adapter'ın yeterli olduğunu iddia etmek değildir.

## Neden bu ilk yol?

- Bu checkpoint/kaynak zaten checksum/revizyon ile doğrulanıp çalıştırıldı; yeni büyük model indirmeden gerçek entegrasyon denetlenebilir.
- Audio semantic branch, sayısal parametreleri metne çevirme veya metin şablonlarına saklama gerektirmez.
- Codec, CLAP ve backbone dondurulur; ilk öğrenilecek bileşen kontrol adapter'ıdır. Çalışma belleği yine backbone'u kapsar; adapter küçük diye tüm eğitim hafif sayılmaz.
- Olay kimliği ile akustik/fiziksel kontroller ayrı kaynaklardır. Aynı event prototype altında kontrol başarısı ayrı ölçülebilir.

AudioLDM'nin önceki tek kuş koşumu kalite üstünlüğü kanıtı değildir. Bu seçim erişim ve teknik entegrasyon içindir; kullanılabilir Foley kalitesi henüz kabul edilmedi.

## Karşılaştırılan yollar

| Yol | Karar gerekçesi | Açık eksik |
|---|---|---|
| AudioLDM m-full + yeni adapter | Native audio conditioning ve çalıştırılmış full checkpoint; ilk entegrasyon | Foley adapter eğitimi, çok-kayıt prototypes, kalite/kontrol kanıtı |
| TangoFlux DiT + yeni kontrol encoder'ları | Kaynakta latent seq645/channel64, joint attention1024 ve açık training yolu; DiT genişleme adayı | Resmî inference T5 text koşullu; prompt-free semantic replacement ve eğitim gerekli; burada inference yok |
| Stable Audio Open / Sketch2Sound / FlashFoley yaklaşımı | Latent transformer + time-varying kontrol açısından güçlü yöntem referansı | İlgili kontrol checkpoint/code erişimi burada doğrulanmış değil; user prompt gerektirmeyen semantik yolu ayrıca çözmek gerekiyor |
| AudioControlNet | Sayısal/temporal control adapter karşılaştırıcısı; önceki resmî yol caption + control | Native prompt-free semantik yol ve fiziksel kontrol genişlemesi doğrulanmadı; burada inference yok |
| TFOLEY | Gerçek sınıf + RMS baseline'ı | 7 sınıf / tek RMS koşulu; hedef çoklu kontrol genişliği yok |
| F-RAVE | Descriptor conditioning / latent ayrıştırma referansı | Genel Foley checkpoint/kalite ve eğitim maliyeti boşluğu |
| FoleyCrafter | Pretrained generatör üstüne modüler semantic/temporal adapter örneği | Resmî yol video-to-audio; video bu projenin girdisi değil |
| PAVAS | Fiziksel parametre conditioning için ilgili yöntem | Video/VLM/trajectory girdileri ve fiziksel etiketler; doğrudan kullanıcı sayısal Foley çözümü olarak uygulanmadı |

## Kaynak denetimi

- [AudioLDM resmî kod](https://github.com/haoheliu/AudioLDM), pin `4054cb418ba3d947d94f6ad302f1d6a320e2313c`: `ldm.py::apply_model`, `q_sample`, `sample_log`; CLAP waveform conditioning. Native model conditioning key **FiLM**, bunu cross-attention diye raporlamıyoruz.
- [TangoFlux kaynak](https://github.com/declare-lab/TangoFlux/blob/164420faa2b3f8a28bb013ccde7006422a616fc7/tangoflux/model.py), pin `164420faa2b3f8a28bb013ccde7006422a616fc7`: FluxTransformer, T5 encoder, duration embedder; `forward` rectified-flow loss. Resmî generator'ı boş text ile çağırmak parametrik kontrol değildir.
- [Sketch2Sound yazar sayfası](https://hugofloresgarcia.art/sketch2sound/): descriptor encoder'ları latent DiT girdisine ekleme; yayımlanmış text + control örnekleri. Kendi adapter'a kaynak fikri, yeniden üretim değil.
- [FlashFoley proje sayfası](https://anonaudiogen.github.io/web/) ve [Sony araştırma açıklaması](https://ai.sony/blog/neurips-2025-sony-ais-latest-contributions): sketch control/acceleration yöntemi. İncelenen sayfalarda çalıştırılabilir resmî kontrol ağırlık linki doğrulanmadı; 'model yok' diye evrensel iddia etmiyoruz.
- [FoleyCrafter resmî kaynak](https://github.com/open-mmlab/FoleyCrafter/blob/main/inference.py): Auffusion, semantic ve temporal adapter, video girdisi.
- [PAVAS makalesi](https://arxiv.org/abs/2512.08282): kütle/hızdan physics adapter ve VGG-Impact değerlendirmesi; fiziksel parametre etiketleme sorusuna referans.

## Bu commit'te doğrulanan / doğrulanmayan

Typed registry, yeni eksik parametre eklenince eski encoded alanın korunması, uygulanamaz kontrolün reddi, sıfır residual başlangıcı ve pretrained gerçek backbone üzerinde optimizer gradient entegrasyonu test edilir. İki optimizer adımı başarı ölçümü değildir. Eksik kontrolün contribution'ının sıfır olması, eğitim sonrası bütün eski ses kalitesinin korunduğunu kanıtlamaz.

Kaliteli kontrollü ses sonucu için: çok-kayıt veri manifesti, adapter eğitimi, ayrı recording test set, ortak-gain dinleme, kategori uyumu, kontrol hatası ve yeni-parametre retention gerekir. Bu aşamalar tamamlanmadan ana modeli başarılı/bitmiş ilan etmiyoruz.

- [AudioControlNet resmî README](https://github.com/juhayna-zh/AudioControlNet): model/control checkpoint ve inference yönergeleri; ilk proje incelemesinde hibrit text + control baseline olarak tutuldu.
