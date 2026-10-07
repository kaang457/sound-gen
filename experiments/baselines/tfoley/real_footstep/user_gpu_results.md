# Yerel GPU koşumu — kullanıcı tarafından paylaşılan sonuç

Kaynak: Kullanıcının 7 Ekim 2026'da yapıştırdığı notebook çıktıları; yerel WAV burada alınmadı veya dinlenmedi. [Ham kayıt](user_gpu_report.json).

| Ölçüm | GPU, kullanıcı koşumu |
|---|---:|
| Torch / cihaz | 2.14.1+cu130 / CUDA |
| Sampling | 100 adım, seed 42, Footstep |
| Süre | 30.656 s |
| Smoothed RMS MAE | 0.005230 |
| Smoothed RMS Pearson | 0.772580 |
| Referans / çıktı RMS | 0.023300 / 0.021729 |
| Referans / çıktı RMS tepe sayısı | 8 / 7 |
| Peak | 0.120898 |

Referansın 1.544 s tepesine en yakın çıktı tepesi 1.283 s; -261 ms. Bu en yakın eşleme başka referans tepesinin kullandığı çıktı tepesini tekrar kullanıyor. Dolayısıyla büyük offset'i bağımsız gecikme ölçüsü diye yorumlamıyoruz: seçili çıkarıcı 1.544 s yakınında bir çıktı tepesi bulmamış. RMS peak'leri onset/gerçek adım sayısı değildir; eşik ve minimum mesafe sonucu etkiler.

CPU koşumuyla aynı kaynak MP3 hash'i, crop, sr ve referans peak zamanları bildirilmiş; global RMS de çok yakın. WAV hash'leri farklı. Bu fark, kaynağın farklı olduğu anlamına gelmez; decoding/resampling/numerik sürümler farklılaşabilir. Neden ayrıca doğrulanmadı. Torch, numpy, librosa ve cihaz farklı olduğundan kalite/kontrol farkını yalnız GPU'ya bağlamıyoruz.

Enerji eğrisi kısmen takip ediliyor; timbre/kategori benzerliği ve kalite bu sayılardan çıkarılamaz. Bu koşum için dinleme değerlendirmesi alınmadı. İlk sentetik deneye verilen geri bildirim bu kayda taşınmadı.
