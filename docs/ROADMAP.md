# Güncel ilerleme planı

## 1. Ana mimari seçimi — ilk entegrasyon kararı verildi

Foley kapsamı ve dinlenebilir temel kalite üzerinden aday üreticileri değerlendirmek. Her aday için kullanıcı prompt'u olmadan olay/koşul verme yolu, ağırlık/lisans, gerçek inference imkânı, genişletilebilir kontrol bağlantısı, veri/eğitim ve donanım gereksinimleri kaydedilecek. İçinde text encoder olması tek başına eleme değildir; yalnız boş prompt denemesi de prompt-free kontrollü üretimi kanıtlamaz.

Teslim: aday tablosu, kaynaklarla destekli mimari kararı, dinlenebilir örnekler ve açık kalan riskler. Uygun hazır model bulunmazsa kendi modelimiz için temel mimari ve veri ihtiyacı seçilecek. Küçük spectral CVAE varsayılan ana model yapılmayacak.

İlk karar [ADR 001](decisions/001_foley_conditioning_architecture.md); AudioLDM + audio prototypes + residual adapter gerçek model üzerinde bağlandı. Ses kalitesi kabulü bekliyor.

## 2. Genişletilebilir koşul tasarımı — ilk kod ve testler mevcut

Olay ailesi + fiziksel/akustik/zaman kontrollerinin ortak şeması; parametre kimlikleri, tipler, birimler, geçerlilik ve eksik-değer maskeleri. Seçilen generator'a koşul aktarımı ve yeni parametre ekleme prosedürü tasarlanacak. Yalnız yeni bir slider eklemek tamamlanmış kontrol sayılmayacak.

Teslim: mimari şema, kontrol sözleşmesi, ilk eğitim kontrolleri ve sonradan eklenecek örnek bir parametre için uyarlama yolu.

Typed registry/encoder bankası, applicability/missing-value kontrolü, yeni encoder ekleme ve epsilon training hattı `src/foley/` içinde. Gerçek modelde iki adım yalnız entegrasyon denetimi.

## 3. Veri ve eğitim protokolü — sıradaki esas aşama

Lisansı açık, yeterli çeşitlilikte Foley kayıtları; kayıt bazında ayrımlar, tutarlı öznitelik çıkarımı, fiziksel metadata ve ortak parametre kapsaması. Kaynak dosyadan çıkarılamayan fiziksel etiketler tahminle gerçek etiket gibi sunulmayacak. GPU üzerinde kısa batch/step deneyiyle bellek ve süre ölçülecek.

Teslim: veri manifesti, split/etiket protokolü, yeniden üretilebilir eğitim ayarları ve ölçülmüş maliyet.

## 4. İlk anlamlı kontrollü model

Az sayıda kontrolle kaliteli Foley üretimini doğrulamak. Parametreli çıktı, kontrolsüz temel çıktı ve hedef davranış birlikte değerlendirilecek. Kontrol öğrenilirken kalite bozulursa karar yeniden ele alınacak.

Teslim: çalışan checkpoint, sade inference notebooku, ortak-gain dinlenebilir örnekler, kontrol/kalite/çeşitlilik sonuçları. Başarısız sonuçlar da kaydedilecek.

## 5. Yeni parametre ve olay ailelerine genişleme

Yeni parametre ekleyip eski kontrollerin korunmasını test etmek; farklı Foley ailelerinde geçerlilik ve genellemeyi değerlendirmek. Bütün parametreleri aynı anda eğitmek zorunlu değil; genişleme tasarımı ilk mimari kararından itibaren gözetilecek.

## Önceki çalışmalar

01–05 ölçüm hazırlığı, 06–07–09 TFOLEY baseline, 08 referans-conditioning, 10 küçük CVAE prototipi. Çalışan kod ve sonuçlar korunur; bu notebookların tamamlanması yukarıdaki aşamaların tamamlandığı anlamına gelmez. Aktif hedef [PROJECT_SCOPE.md](PROJECT_SCOPE.md).

[Veri protokolü](FOLEY_DATA_PROTOCOL.md), cache/prototype/eğitim kodları hazır. Tek kayıtlı cache tanısı production dataset değildir; çok-kayıt eğitim ve validation/test sonuçları henüz yok.
