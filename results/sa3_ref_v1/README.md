# sa3_ref_v1: Stable Audio 3 ile oyun sesi referans seti

Hazır model, eğitim yok: **Stable Audio 3 Small-SFX** (Stability AI). Sesleri dinlemek için [index.html](index.html) dosyasını indirip tarayıcıda aç (her sesin tüm varyasyonları çalınabilir); her sesin 1. varyasyonu WAV olarak [`sa3_ref_v1_v1.zip`](./sa3_ref_v1_v1.zip) içinde.

| | |
|---|---|
| Model | `stabilityai/stable-audio-3-small-sfx` (kod commit `77e0398`) |
| Ayar | 8 adım, 44100 Hz stereo, istek başı 4 varyasyon, tohum 1, istek öneki `TrackType: SFX,` |
| GPU | NVIDIA A100-SXM4-40GB |
| Lisans | Stability AI Community License: yıllık geliri 1 M $ altı kuruluşlara ticari kullanım dahil serbest, çıktılar kullanıcının; üstü kurumsal lisans. Metin kodlayıcı Gemma kullanım şartlarıyla. Ürüne girmeden lisans metni okunmalı. |
| Eğitim verisi (Stability) | ~1,28 M kayıt: AudioSparx (lisanslı) + Freesound CC0 / CC-BY / Sampling+ |
| İstek eşleşmesi (CLAP) | kendi isteği 32 istek arasında 1. sırada %48, ilk 3'te %60 |
| Tek / çok olay | tek tabanca atışı: [6, 10, 7, 7] · çatışma: [20, 24, 21, 17] (varyasyon başına sayılan olay) |

| # | Kategori | Ses | İstek | Süre | Üretim | Tepe | CLAP sırası (varyasyonlar) |
|---|---|---|---|---|---|---|---|
| 1 | Silah | tek tabanca atışı | A single pistol gunshot, sharp crack with a short punchy tail, close mic, outdoors | 2.5 s | 0.62 s | -8.4 dBFS | 24 19 26 29 |
| 2 | Silah | çatışma | A gunfight: rapid bursts of automatic rifle fire and scattered pistol shots from several shooters, outdoor battlefield | 6 s | 0.58 s | 0.0 dBFS | 1 1 1 1 |
| 3 | Silah | pompalı tüfek | A pump-action shotgun: pump rack followed by one heavy shotgun blast with a boomy tail | 3 s | 0.65 s | -5.5 dBFS | 4 6 4 4 |
| 4 | Silah | lazer silah | A sci-fi laser gun shot, bright zappy pew with a fast pitch drop and a short electric tail | 2 s | 0.63 s | -2.6 dBFS | 1 1 1 1 |
| 5 | Silah | kılıç çarpışması | Two metal swords clash, bright ringing steel impact with a short metallic shimmer | 2.5 s | 0.66 s | -6.0 dBFS | 24 28 25 26 |
| 6 | Silah | ok ve yay | A bow releases an arrow: string twang, fast whoosh, the arrow thuds into wood | 2.5 s | 0.53 s | -5.7 dBFS | 19 21 19 19 |
| 7 | Patlama / darbe | büyük patlama | A large explosion, deep boom with debris falling and a long rumbling tail | 5 s | 0.63 s | 0.0 dBFS | 1 1 1 1 |
| 8 | Patlama / darbe | uzak patlama | A distant explosion heard from far away, muffled low boom echoing across a valley | 5 s | 0.55 s | 0.0 dBFS | 2 2 2 2 |
| 9 | Patlama / darbe | yumruk | A heavy punch impact, meaty thud with a fast cloth whoosh, fighting game | 1.5 s | 0.52 s | 0.0 dBFS | 3 15 6 20 |
| 10 | Patlama / darbe | cam kırılması | A window pane shatters, glass breaking and shards falling on a hard floor | 3 s | 0.53 s | 0.0 dBFS | 1 1 1 1 |
| 11 | Bilim kurgu | uzay gemisi motoru | A large spaceship engine idling, deep steady sci-fi hum with soft mechanical whir, loopable | 6 s | 0.58 s | -11.4 dBFS | 1 1 1 1 |
| 12 | Bilim kurgu | uzay gemisi geçişi | A sci-fi spaceship flies past fast from left to right, roaring jet whoosh with doppler pitch shift | 5 s | 0.58 s | 0.0 dBFS | 1 1 1 1 |
| 13 | Bilim kurgu | enerji kalkanı | A sci-fi energy shield powering up, rising electric hum with a crisp shimmering lock-in | 3 s | 0.53 s | -3.7 dBFS | 9 5 6 4 |
| 14 | Bilim kurgu | ışınlanma | A sci-fi teleport effect, swirling sparkly whoosh ending in a soft electric pop | 3 s | 0.55 s | -9.8 dBFS | 1 2 1 2 |
| 15 | Yaratık | zombi hırıltısı | A zombie growl, wet raspy throaty groan, slow and menacing, close mic | 4 s | 0.56 s | -6.2 dBFS | 2 2 2 2 |
| 16 | Yaratık | canavar kükremesi | A huge monster roar, deep guttural bellow with a gritty distorted edge | 4 s | 0.57 s | -0.2 dBFS | 1 1 1 1 |
| 17 | Yaratık | ejderha kanatları | A dragon flapping its huge leathery wings, heavy slow whooshes of air | 4 s | 0.60 s | -1.2 dBFS | 1 1 1 1 |
| 18 | Yaratık | köpek havlaması | A medium-size dog barking several times, outdoors | 4 s | 0.58 s | -0.6 dBFS | 1 1 1 1 |
| 19 | Hareket / foley | çakılda ayak sesi | Footsteps walking on gravel, steady pace, crunchy close-mic foley | 5 s | 0.54 s | -8.8 dBFS | 1 1 1 1 |
| 20 | Hareket / foley | ahşapta ayak sesi | Footsteps walking on a creaky old wooden floor indoors, boots | 5 s | 0.52 s | -8.2 dBFS | 1 1 1 1 |
| 21 | Hareket / foley | at dörtnala | A horse galloping on a dirt road, rhythmic hoof beats passing by | 5 s | 0.59 s | -1.1 dBFS | 1 1 1 3 |
| 22 | Hareket / foley | kapı gıcırtısı | An old wooden door slowly creaks open with a long squeaky hinge | 4 s | 0.53 s | -18.7 dBFS | 1 1 1 1 |
| 23 | Hareket / foley | sandık açılışı | A heavy wooden treasure chest unlocks with a metal latch click and the lid creaks open | 3 s | 0.54 s | -3.7 dBFS | 1 1 1 1 |
| 24 | Büyü / arayüz | ateş topu büyüsü | A fireball spell: whooshing burst of fire launched forward with a crackling roar | 3 s | 0.53 s | 0.0 dBFS | 6 5 6 5 |
| 25 | Büyü / arayüz | iyileştirme büyüsü | A healing magic spell, soft glittering chimes rising with a warm shimmer | 3 s | 0.56 s | -7.2 dBFS | 16 16 16 17 |
| 26 | Büyü / arayüz | menü tıklaması | A clean short user interface button click, soft and modern | 1 s | 0.52 s | -8.8 dBFS | 6 3 8 15 |
| 27 | Büyü / arayüz | altın toplama | A coin pickup sound for a video game, bright short metallic ching | 1 s | 0.51 s | -7.8 dBFS | 28 28 23 29 |
| 28 | Büyü / arayüz | güçlendirme | A video game power-up pickup, quick rising synth arpeggio with sparkle | 2 s | 0.52 s | -6.7 dBFS | 6 7 7 7 |
| 29 | Ortam | teneke çatıda yağmur | Heavy rain falling on a tin roof, steady dense patter, loopable ambience | 8 s | 0.58 s | 0.0 dBFS | 1 1 1 1 |
| 30 | Ortam | orman | Forest ambience in the morning, birds chirping, light breeze in the leaves | 8 s | 0.69 s | -12.2 dBFS | 5 7 7 7 |
| 31 | Ortam | mağarada rüzgâr | Wind howling through a dark cave, hollow low drone with distant water drips | 8 s | 0.57 s | -3.1 dBFS | 4 3 5 2 |
| 32 | Ortam | kamp ateşi | A crackling campfire, wood pops and soft flames, close mic | 6 s | 0.63 s | -2.0 dBFS | 5 7 3 7 |

Üretim süresi: o sesin bütün varyasyonları birlikte (tek GPU çağrısı). CLAP sırası 1 = ses, bütün istekler arasında en çok kendi isteğine benziyor (ölçüm ayrı bir model, dinlemenin yerini tutmaz). Tepe: 1. varyasyonun en yüksek seviyesi.
