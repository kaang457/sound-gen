# Önceki parametrik deneylerin değerlendirmesi — 2026-10-07

> Güncel yön: [parametrik Foley kapsamı](../../docs/PROJECT_SCOPE.md) ve [ilerleme planı](../../docs/ROADMAP.md). Bu not önceki deneyleri/kod denetimini belgeliyor; TFOLEY, AudioLDM, küçük CVAE veya F-RAVE ana üretim modeli olarak seçilmiş değildir. Seçim, Foley kalitesi ve yeni parametre ekleme kapasitesi üzerinden yapılacak.


Hedef: kullanıcı metni olmadan sayısal özniteliklerle deep learning çıktısını kontrol etmek. Referans sesin semantik embedding'i ile üretim ayrı bir yetenektir.

| Model | Girdi gerçekten decoder/model koşulu mu? | Çalıştırılan kapsam | Karar |
|---|---|---|---|
| TFOLEY | Sınıf + doğrudan 690 karelik RMS | Pretrained 4 s / 22050 Hz; 3 eğri × 2 seed | Enerji/zaman baseline. Pitch/centroid desteklenmez. |
| Repo spectral CVAE | Log RMS + magnitude centroid + iki boyutlu z | Gerçek 4 s Footstep kaydında 3000 eğitim adımı; 12 üretim | Küçük, çalışan iki-parametre kontrol prototipi. Doğal Foley/timbre başarısı doğrulanmadı. |
| F-RAVE | Descriptor conditioning ve latent discriminator | Kod/config denetlendi, model burada eğitilmedi | Descriptor conditioning için mimari referans; ana model seçilmedi. |
| AudioLDM m-full | CLAP referans-audio embedding | İlk kuş inference tamamlandı | Referans içeriği baseline; bağımsız RMS/centroid kontrolü sunan çözüm olarak seçilmedi. |

## F-RAVE kod denetimi

Resmî repo [neurorave/neurorave](https://github.com/neurorave/neurorave), incelenen tree `0b754f7a74c520e6a9b0a653d5f8e0fba070c3f0`. Root README eski olsa da `/code` kaynak içerir. `code/requirements.txt` boş; `code/README.md` Python 3.9 ve eski torch yönergeleri içerir. Dolayısıyla mevcut CUDA ortamına doğrudan kurulabilir bir paket varsayılmadı.

[Config](https://github.com/neurorave/neurorave/blob/0b754f7a74c520e6a9b0a653d5f8e0fba070c3f0/code/config/fader_config.yaml): 16000 Hz, 65536 örnek, batch 8, latent 128; centroid/rms/bandwidth/sharpness/booming; warmup 1,000,000, max_steps 3,000,000. Bunlar kaynak varsayılanlarıdır, donanım süre tahmini değildir. Uygun çoklu kontrollü Foley checkpoint'i doğrulanmadı. Repodaki `model_nsynth_monoattr_CVAE.pth`/`VAE.pth` dosyaları isim/kapsam itibarıyla çoklu F-RAVE Foley sonucu yerine kullanılamaz.

## Bizim CVAE ile makale ilişkisi

[Continuous descriptor-based control for deep audio synthesis](https://arxiv.org/abs/2302.13542) sayısal descriptor conditioning için yöntem referansıdır. Bizim MLP spectral CVAE, F-RAVE'nin waveform generator/PQMF/adversarial reconstruction/latent discriminator mimarisini gerçeklemez. Kendi küçük kontrol deneyimizdir. Koşullu VAE prensibi için [Sohn et al., 2015](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html); VAE için [Kingma & Welling](https://arxiv.org/abs/1312.6114).

İlk denemede prior'dan örneklemede RMS hatası ~%85 çıktı. Posterior kontrol kaybına ek olarak prior kontrol kaybı ve daha güçlü KL ile ikinci eğitim yapıldı. İlk sonuç `initial_trial.json` içinde korunur. Geliştirme tanıları kullanıldığı için son zaman bölümünü bağımsız test benchmark'ı olarak sunmuyoruz.

## Bir sonraki ölçekleme eşiği

Tek kayıtla kalite/genelleme seçimi yapılamaz. Sonraki eğitim için çoklu, lisansı açık Footstep kayıtları ve kayıt bazında train/validation/test ayrımı gerekir; aynı kaydın kareleri iki split'e dağılmamalıdır. Öznitelikler yalnız train istatistikleriyle ölçeklenmeli, sessiz bloklar maskelenmeli ve ortak RMS/centroid kapsaması denetlenmelidir. Bu, küçük deneyin olası devamıdır; aktif projede önce Foley kalitesi ve genişletilebilirlik üzerinden ana mimari seçimi yapılır. GPU batch/step süresi ölçülmeden eğitim süresi verilmez.

Kabul protokolü: ayrı kayıtlarda hedef hata, üçten fazla seed, sabit diğer kontrol altında değişim, ortak-gain dinleme, blok sınırı/artefakt ve ses sınıfı değerlendirmesi. Bu küçük deney yalnız sınırlı iki-kontrol uygulamasını gösterir; güncel ana modelin seçimi veya genişletilebilir Foley hedefinin karşılanması değildir.
