# Notebook'lar

Kendi modelimiz: **01 → 02 → 03**. Hazır modeller: **04** (referans seti + stüdyo), **05** (videodan ses); ikisi de tek başına. Her notebook'un hücre 0'ı repoyu çeker, paketi kurar ve Drive'ı bağlar.

```
01_data ──► pack48_poc (Drive) ──► 02_train ──► features/ + runs/<RUN>/ema.pt (Drive) ──► 03_studio (paylaşım linki)
                                        └──► results/<RUN>/ (GitHub: test sayfası, sesler, ölçümler)
```

## 01_data — veri
`DATASET = "poc_max"`: DCASE + ESC-50 + FSD50K dev + eval, lisans filtresiz (yalnız kavram kanıtı), 48 kHz,
≤ 6 s parçalar. Drive'da ~16–17 GB. `commercial`: yalnız CC0 / CC BY.

## 02_train — model
| Hücre | Ne yapar |
|---|---|
| 1 | ayarlar (`RUN`, `PACK`, `MAX_HOURS`, `MODEL`, `BATCH`); MMAudio kodu (sabit sürüm) + 44,1 kHz VAE (Drive'a bir kez) |
| 2 | her klip için açıklama (FSD50K'da etiketler + Freesound başlığı), VAE latentleri, CLIP metin token'ları; Drive'a kaydedilir; **yeniden çalma testi**: gerçek klip → VAE → geri açılmış ses (kalitenin üst sınırı) |
| 3 | kuru deneme ~5 dk: kayıp düşüyor mu, NaN yok mu, bir istekten 8 aday |
| 4 | eğitim (`MAX_HOURS`), ardından test seti: 20 istek × 8 aday; ezber testi, CLAP, tek/çok olay çifti; sayfa + zip → GitHub `results/<RUN>/` |
| 5 | kayıp eğrisi + test sayfası |

## 03_studio — GUI
Tek hücre: gradio + model + VAE/BigVGAN; `…gradio.live` bağlantısı. İstek → 8 aday (A, B, C + A→B %33, A→B %67,
B×C, A×C, A×B×C) → efekt rafı → mono / stereo WAV ya da hepsi zip.

## 04_stable_audio — hazır model (eğitim yok)
Stable Audio 3 Small-SFX (Stability AI, açık ağırlık). Gerekli: Hugging Face'te iki sayfada lisans onayı
([model](https://huggingface.co/stabilityai/stable-audio-3-small-sfx), [T5Gemma](https://huggingface.co/google/t5gemma-b-b-ul2))
ve Colab secret `HF_TOKEN`. Ağırlıklar bir kez iner (yerel disk, Drive'da `hf_cache.tar` yedeği); `OFFLINE = True` internetsiz yükler.

| Hücre | Ne yapar |
|---|---|
| 1 | kurulum (Colab'ın torch'una dokunmadan), erişim kontrolü, model yükleme, tek ses denemesi |
| 2 | referans seti: 32 oyun sesi (silah, patlama, bilim kurgu, yaratık, foley, büyü / arayüz, ortam) × 4 varyasyon; CLAP istek eşleşmesi, tek / çok olay; sayfa + README tablosu + WAV zip → GitHub `results/<RUN>/`; tüm varyasyonlar Drive'da |
| 3 | sayfayı notebook'ta göster |
| 4 | kendi isteklerin (`istek \| saniye`) |
| 5 | stüdyo aynı modelle: 8 varyasyon + efekt rafı + mono / stereo WAV |

## 05_video_to_audio — sessiz video → sesli video (eğitim yok)
Qwen3-VL (Apache-2.0) + Stable Audio 3 Small-SFX. GPU: L4 / A100 (8B planlayıcı), T4'te 4B. 04'teki HF hazırlığı yeterli.

| Hücre | Ne yapar |
|---|---|
| 1 | modeller (yerel önbellek + Drive yedeği, `OFFLINE`), planlayıcı boyutu GPU belleğine göre |
| 2 | **video stüdyosu (arayüz):** video yükle → sahneyi çöz → sesli video + zaman çizelgesi; her iz için ses (dB), sol ↔ sağ, yakın → uzak, sustur, solo, zaman, istek → yeniden üret, sil, tek başına dinle; iz ekle; paket indir |
| 3–7 | aynı işler arayüzsüz, kodla (plan, düzeltme, üretim, tek iz değiştirme, indirme) |

Ses seviyesi / konum / sustur / solo model çağırmaz (~1 s, üretilen ses önbellekte). Zaman ve istek değişikliği yalnız o izi yeniden üretir.
Çıktılar Drive'da `MyDrive/sound-gen/v2a/<video adı>/`. Videolar GitHub'a gönderilmez (repo herkese açık).
