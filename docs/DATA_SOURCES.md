# Ticari model eğitimi için ses verisi kaynakları (Ekim 2026)

Amaç: Stable Audio 3'e (ve ileride kendi modelimize) ince ayar yapmak için **ticari kullanıma ve model eğitimine izin
veren** ses verisi. Otomatik indirilebilenler `notebooks/06_licensed_data.ipynb` ile iner (`soundgen/sources.py`).
Hukuki görüş değildir: ürüne girmeden her kaynağın lisans metni okunmalı, `manifest.csv` atıf listesi olarak saklanmalı.

**Tutulan lisanslar:** CC0, kamu malı (PD), CC-BY 3.0 / 4.0, OGA-BY.
**Atılanlar:** NC (ticari olmayan), ND (türev yasak), SA (aynı lisansla paylaş; model ağırlıkları için belirsiz),
Sampling+, GPL, lisansı bilinmeyen.

## 1. Otomatik (06 notebook)

| Kaynak | İçerik | Lisans | Ölçek | Not |
|---|---|---|---|---|
| Freesound (HF: `benjamin-paine/freesound-laion-640k`) | ~506 bin Freesound klibi, klip başına lisans ve "ticari kullanım" alanı | CC0 / CC-BY süzülür | toplam 679 GB; varsayılan 40 GB, oyunla ilgili kelimeler | Stable Audio Open da aynı havuzun CC0 / CC-BY kısmıyla eğitildi (266 bin CC0 + 195 bin CC-BY). Üçüncü taraf yüklemesi: lisans Freesound meta verisinden |
| Clotho v2 (Zenodo) | ~7 bin Freesound klibi, her birine 5 insan yazımı açıklama | dosya başına; CC0 / CC-BY süzülür | 6,5 GB | açıklama kalitesi en yüksek kaynak |
| OpenGameArt | oyun ses efekti paketleri | CC0 / CC-BY / OGA-BY süzülür | binlerce paket; varsayılan 400 öğe | sayfalar yavaş ve kibar taranır |
| Kenney.nl | darbe, arayüz, RPG, bilim kurgu, dijital, kumarhane ses paketleri | CC0 | < 1 GB | oyun sesi için çok temiz |
| Wikimedia Commons | efekt, hayvan, araç, silah, patlama, hava, su, makine, çan, tren, uçak, gemi, ayak sesi kategorileri | CC0 / PD / CC-BY süzülür (çoğu CC-BY-SA, atılıyor) | varsayılan 5.000 dosya | kalite değişken |
| iNaturalist (API) | kuş, memeli, amfibi, böcek sesleri | ses lisansı CC0 / CC-BY | taksonda 1.500 | API ~1 istek / s |
| Blender açık filmleri | Big Buck Bunny, Elephants Dream, Sintel, Tears of Steel (görüntü + ses) | CC-BY | 4 film | ses eğitimi için değil (müzik + diyalog): 05'i denemek ve ölçmek için |

## 2. Elle / satın alma

| Kaynak | Ne | Lisans | Nasıl |
|---|---|---|---|
| Pro Sound Effects (AI eğitim seti) | 1,27 M ses, 5.000+ saat | ticari eğitim lisansı | teklif |
| BOOM Library | sinematik ve oyun sesleri | eğitim lisansı ayrıca | teklif |
| Shutterstock (ses) | büyük SFX kütüphanesi | eğitim lisansı ayrıca | teklif |
| Troveo | lisanslı gameplay video verisi | eğitim lisansı | teklif (video tarafı) |
| Freesound API (orijinal dosyalar) | yukarıdaki havuzun kayıpsız asılları | dosya başına | API anahtarı + OAuth2; günlük sınır |
| NASA ses arşivi | roket, görev kontrol, uzay araçları | kamu malı (ABD hükümeti) | nasa.gov / SoundCloud, elle |
| Kendi foley kayıtlarımız | ayak, kumaş, metal, kapı, darbe | bizim | mikrofon + sessiz oda; en temiz yol |
| Kendi oyun yakalamalarımız | Unreal sahneleri, lisanslı ses paketleriyle | paketin lisansı | paketin "AI eğitimi" iznine bakılmalı |
| Oyun stüdyoları | stüdyonun kendi sesleri | stüdyo ile anlaşma | karşılığında o oyuna özel model |
| Kamu malı film / çizgi film | 1929 ve öncesi sesli yapımlar (ABD'de 2025'te Hitchcock'un *Blackmail*'i gibi) | kamu malı; başlık başlık doğrulanmalı | ses müzik + diyalog + efekt karışık: efekti ayırmak için ayrıştırma modeli gerekir; restore edilmiş sürümlerin ve karakter markalarının hakları ayrı; düşük öncelik |

## 3. Kullanılmayacaklar (neden)

| Kaynak | Neden |
|---|---|
| Pixabay | Hizmet şartları: "makine öğrenmesi amaçlı" otomatik veri toplama ve toplu kopyalama yasak |
| Sonniss GDC paketleri | lisans: yapay zekâ modeli eğitimi yasak |
| BBC Sound Effects | RemArc lisansı: ticari olmayan |
| iNatSounds (hazır set) | kullanım şartları: ticari olmayan araştırma; aynı sesler tek tek iNaturalist API'den lisansa göre alınabilir (yukarıda) |
| ESC-50, UrbanSound8K | CC-BY-NC |
| AudioSet, VGGSound, AudioCaps | YouTube videoları; lisans yok |
| WavCaps | akademik kullanım; BBC ve AudioSet içerir |
| FSD50K / Clotho'nun NC klipleri | süzülüyor; yalnız CC0 / CC-BY kısmı kullanılıyor |
| Zapsplat, Mixkit vb. ücretsiz siteler | lisansları kütüphane dışı kullanım ve yeniden dağıtımı kısıtlıyor; AI eğitimi izni yok sayılmalı |
| Oyunlardan / filmlerden kayıt | yayıncının telif hakkı; kullanıcı sözleşmeleri içerik çıkarmayı yasaklıyor |

## 4. Konuşma (ileride)

Common Voice (CC0), LibriTTS-R (CC-BY 4.0), MLS (CC-BY 4.0). Ses efektinden ayrı model ve ayrı iş kolu.
