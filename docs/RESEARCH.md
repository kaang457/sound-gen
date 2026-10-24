# Araştırma: buraya nasıl gelindi (Ekim 2026)

Bütün kod, notebook'lar, sonuçlar ve ayrıntılı karar günlüğü (`docs/PROCESS.md`) [`research-v1`](https://github.com/kaang457/sound-gen/tree/research-v1)
dalında; eski çıktılar `archive` dalında. İş sahibi için kısa özet: [Ses Üretimi: Şimdiye Kadar Ne Yaptık](https://claude.ai/artifact/E49GGq7FiYGD89j91cPWST).

| Adım | Ne denendi | Sonuç | Ders |
|---|---|---|---|
| base_v1 | WaveFlow (sıfırdan yazıldı) + 20 bantlık mel taslak + kategori; DCASE + ESC-50 | 1378 Hz katlarında vızıltı, taslağı takip etmiyor | sağlık raporu: koşul yolu sese göre 10–45 kat zayıf |
| düzeltmeler | taslak normalizasyonu, ActNorm, interp upsampler | vızıltı yok, koşul oranı 0,16 → 0,70 | ölçmeden düzeltme yapma |
| poc_v1 | 48 saat lisans filtresiz veri, 81 bin adım | seviye ±2 dB, zarf korelasyonu 0,84–1,00; ama ses referans kayıttan geliyor | sıcaklık 0,8 sesi 7–9 dB kısıyordu; 1,0 doğru |
| kontroller | uzaklık, ortam, şiddet, süre, olay sayısı (mel üzerinde, kural tabanlı) | model düzenlemeleri tutturdu (seviye ±1,6 dB, yankı 1,31 / 1,38 s) | kural tabanlı olay çoğaltma istenmedi: "çatışma"yı model anlamalı |
| sıfırdan üretim | vektör → taslak diffusion → WaveFlow; ezber testi | altyapı kuruldu, büyük eğitime geçilmedi | 20 bant ayrıntı için yetersiz |
| mel sözleşmesi | 48 kHz / 128 bant, sonra BigVGAN-v2 ile birebir 44,1 kHz / 128 / 512 | BigVGAN'ın mel'iyle fark 7e-7 | MMAudio da tam bu tanımı kullanıyor |
| CLAP koşulu | etiket yerine CLAP ses vektörü (AudioLDM yöntemi) | eğitim başlamadan mimari değişti | istek = metin token dizisi daha güçlü |
| stüdyo | Gradio, varyasyonlar + gürültü karışımları, DAW efekt rafı (sentetik IR) | çalışıyor, yeni mimariye taşındı | efektler sesin üzerinde: kesin ve anında |

**Yeni mimariye taşınanlar:** mel tanımı (MMAudio ile aynı), efekt rafı, stüdyo arayüzü, ezber testi, CLAP
tanınabilirlik testi, olay sayımı, veri indirme ve paketleme, Colab altyapısı (Drive, GitHub'a gönderim, devam etme).
