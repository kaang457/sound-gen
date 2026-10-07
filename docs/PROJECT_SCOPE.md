# Proje kapsamı — 2026-10-07

## Amaç

Çoğunlukla Foley seslerine yönelik, kullanıcı metin prompt'u gerektirmeyen, genişletilebilir parametrik deep learning ses üretimi. Sistem kaliteli ve olayla tutarlı ses üretirken çok sayıda kontrolü zaman içinde ekleyebileceğimiz bir yapıya sahip olmalı. Mevcut küçük deneyler ana çözüm olarak kabul edilmez.

## Kontrol sözleşmesi

| Grup | Hedef örnekleri | Veri gereksinimi |
|---|---|---|
| Olay / fiziksel bağlam | Foley ailesi, malzeme, yüzey, kuvvet, boyut, hız | Kaynak metadata'sı, etiket veya kontrollü kayıt; anlamlı parametre tanımı |
| Akustik | RMS/şiddet, centroid/parlaklık, rezonans, sönüm, uygun olaylarda pitch | Tutarlı öznitelik çıkarımı; fiziksel parametrelerle özdeş sayılmamalı |
| Zaman | Başlangıç, süre, tekrar, kontrol eğrileri | Zaman hizalı etiket/öznitelik veya kontrollü olay dizileri |

Bir parametre tanımında isim, tip (sayısal/kategorik/eğri), birim, aralık, uygulanabildiği ses ailesi ve eksik-değer davranışı bulunmalı. Eksik kontrol sıfırla eşanlamlı olmamalı; uygulanamaz kontrol sessizce yok sayılmamalı. Bunlar tasarım gereksinimleridir, henüz uygulanan ortak API yetenekleri değildir.

Tek bir sabit iki-boyutlu vektöre bağlı mimari hedefi karşılamaz. Kontrol encoder'ı, parametre kimliği/değeri, zaman yapısı ve varlık/geçerlilik maskelerini temsil edebilmeli; temel üreticiye koşul olarak aktarılabilmelidir. Tam yöntem (token, modüler encoder, adapter vb.) model seçimi sırasında karşılaştırılacak; henüz kararlaştırılmadı. Yeni bir parametre için eğitim/uyarlama gerekebilir; sıfır eğitimle sınırsız kontrol vaadi yoktur.

## Başarı ölçütleri

1. Çıktı dinlendiğinde hedef Foley olayına uygun ve kullanılabilir kalitede olmalı; kontrolsüz temel modelin kalitesi de incelenmeli.
2. Aynı diğer koşullar/seed altında parametre değişimi beklenen duyulabilir ve ölçülebilir etkiyi üretmeli.
3. Bir kontrol değişirken diğer özniteliklerdeki etki raporlanmalı. Fiziksel bağımlılıklar nedeniyle tam bağımsızlık her parametre için varsayılmamalı.
4. Birden fazla kayıt/örnek/seed ile değerlendirme ve kayıt bazında train/validation/test ayrımı yapılmalı.
5. Yeni kontrol ekleme yolu, veri ihtiyacı ve mevcut kontrollerin korunması somutlaştırılmalı.

Başlangıç doğrulaması birkaç kontrol ve bir Foley ailesiyle yapılabilir. Bu, nihai kapsamı o aileye daraltma veya genişletilebilirliği erteleme gerekçesi değildir. Kesin metrik eşikleri değerlendirme protokolü hazırlanırken belirlenecek; mevcut küçük denemelerden türetilmiş evrensel eşik yoktur.

## Kapsam dışında / yardımcı

Genel müzik, video-to-audio, yalnız referans kopyalama ve sırf sayısal öznitelik tutturmak ana hedef değildir. Referans-audio conditioning ve DSP/ölçüm çalışmaları yardımcı olabilir. Çıkışa gain/filtre uygulamak learned model control ile ayrı raporlanır.

## Bugünkü durum

TFOLEY temporal baseline; AudioLDM referans-conditioning baseline ve ilk adapter-backbone entegrasyonu; tek-kayıt spectral CVAE küçük mühendislik deneyi; F-RAVE mimari referans. Hiçbiri bütün gereksinimleri karşılayan ana model olarak seçilmedi. Yeni çalışma bu kapsam ve model seçim planına göre ilerlemeli.

## İlk mimari entegrasyonu

[ADR 001](decisions/001_foley_conditioning_architecture.md): AudioLDM m-full + audio prototype bankası + modüler epsilon residual kontrol adapter’ı. Scalar/curve/categorical schema ve gerçek pretrained gradient bağlantısı var; çok-kayıt adapter eğitimi ve kalite/kontrol değerlendirmesi yok. Ana modelin başarılı kabulü bu kararın tamamlandığı anlamına gelmez.
