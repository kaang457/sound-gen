# Referans benzerliği ve sayısal kontrol için güncellenen seçim

Kontrol tarihi: 2026-10-07. İlk dinlemede kullanıcı beklenen benzerliği bulmadı. Bu nedenle üç ayrı hedef için ayrı aday tutuluyor; tek bir modeli bütün ihtiyaçlara uygun kabul etmiyoruz.

| Hedef | Öncelik | Kanıt / sınır |
|---|---|---|
| Gerçek Footstep'in RMS ve zaman yapısını takip | T-FOLEY, notebook 07 | Metinsiz class + RMS; kuş sınıfı yok, timbre taklidi yok |
| Referans kuş/çevresel kayıtla benzer içerik üretme | AudioLDM audio-to-audio, sonraki inference adayı | Resmî kodda waveform → CLAP audio embedding yolu var; bu ortamda henüz çalıştırılmadı |
| Sayısal öznitelikleri bağımsız değiştirme | F-RAVE yöntem incelemesi | Metinsiz descriptor conditioning; uygun genel Foley checkpoint doğrulanmadı, published çoklu kontrol skorları sınırlı |

## AudioLDM kaynak denetimi

[Resmî README](https://github.com/haoheliu/AudioLDM) audio-to-audio modunu belgeliyor. İncelenen revizyon: `4054cb418ba3d947d94f6ad302f1d6a320e2313c`. `audioldm/pipeline.py` referans WAV verildiğinde `set_cond_audio` ile condition key'i waveform ve embed mode'u audio yapıyor. Yani yalnız boş caption ile text koşullu üretim değil.

Önemli ayrıntı: `audioldm/ldm.py::generate_sample`, birden çok aday olduğunda çıktı ile text arasında CLAP benzerliğiyle seçim yapıyor. Audio-to-audio ilk deneyde **batchsize 1 ve candidate 1** kullanılmalı; bu text-ranking dalı çalışmaz. CLAP text branch ve empty unconditional embedding modelin içinde hâlâ bulunur. Dolayısıyla kullanıcı prompt'u gerektirmeyen inference, tüm text bileşenlerinin mimariden kaldırılması demek değildir.

`audioldm-m-full`/`s-full` aileleri audio-conditioned checkpoint olarak belgeleniyor; `text-ft` farklıdır. Diffusers'ın standart text pipeline'ına boş prompt vermek, bu audio yoluyla aynı deney değildir. Eski bağımlılıklar için ayrı ortam gerekir; mevcut T-FOLEY/CUDA ortamını değiştirmeden önce uyumluluk denetlenmeli.

## Araştırma yorumu

AudioLDM bu kez referansın enerji zarfıyla sınırlı kalmadan içerik embedding'ini kullanıyor; bu nedenle kuş referansı hedefi için sıradaki aday. Birebir waveform/timbre taklidi, kuş türü doğruluğu veya onset korunması garanti değil. RMS/pitch/brightness için bağımsız slider kontrolü de bu adayla çözülmüş sayılmaz.

[DarkGAN](https://arxiv.org/abs/2108.01216) metinsiz semantic soft-label yaklaşımı açısından incelendi; çalışma müzikal ses sentezinde değerlendiriliyor, genel çevresel ses/checkpoint başarısı doğrulanmadığı için ana aday yapılmadı. F-RAVE için de alan dışı genelleme aynı şekilde doğrulanmalıdır.

## Sonraki kabul ölçütü

Aynı kuş/Footstep referanslarıyla audio-to-audio; sabit seed, tek aday, caption yok. Referans ve çıktıyı etiketli player'da dinleme, kategori/içerik uyumu ve temporal sapma. Bunun ardından sayısal kontrol için ayrı sweep gerekir. Bu not kod inceleme sonucudur; AudioLDM inference başarısı veya benchmark raporu değildir.
