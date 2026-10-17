# samples

Colab notebook'larının gönderdiği küçük ses dosyaları (22.05 kHz, 16-bit, ≤6 s).

- `references/` — veri setinden etiket başına bir referans (yalnız paylaşılabilir lisanslılar)
- `<RUN>/step_XXXXXX/` — eğitim sırasında: `*_ref.wav` gerçek, `*_gen.wav` aynı taslaktan üretilen
- `<RUN>/gallery/` — 04: her kategori için `_ref`, `_gen0/_gen1`, parametreler (`.json`) ve `metrics.json`
- `ft_*/` — 03: base ve fine-tune modellerin aynı taslak ve aynı tohumla karşılaştırması
