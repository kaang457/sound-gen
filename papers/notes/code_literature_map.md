# Code–literature map

Last checked: 2026-10-07. This is a preliminary relationship map, not a completed literature review.

## 01_audio_basics.ipynb

**Status:** Original educational signal-processing notebook. No paper architecture, training objective, pretrained model or paper-specific evaluation is implemented. The demo sound is synthetic, not model-generated Foley.

| Related work | Relevant idea from the source | Connection to our current code | Still missing |
|---|---|---|---|
| [Sketch2Sound](https://arxiv.org/abs/2412.08550) | Time-varying loudness, brightness and pitch controls | Waveform, time-frequency visualization and RMS are preparatory tools for later control analysis | Exact control extractors, model inference, brightness/pitch measurements, control evaluation |
| [Audio ControlNet](https://arxiv.org/abs/2602.04680) | Loudness, pitch and event-roll conditioning | Audio loading and level checks prepare later inspection of controlled outputs | ControlNet/Adapter, event-roll extraction, inference and benchmark |
| [T-FOLEY](https://arxiv.org/abs/2401.09294) | Sound-class and temporal-event conditioning in waveform generation | Three known demo onsets illustrate event timing on a waveform | The paper's temporal feature, Block-FiLM, diffusion model and evaluation |
| [Audio Generation with Multiple Conditional Diffusion Model](https://doi.org/10.1609/aaai.v38i16.29773) | Multiple audio-generation conditions | Basic audio representations can support later input/output inspection | Paper-specific representations, conditioning modules, inference and evaluation |
| [PicoAudio](https://github.com/zeyuxie29/PicoAudio) | Temporal control of audio events | Known onset times give a basic temporal visualization example | Structured prompts, generation pipeline and timing evaluation |
| [Make-An-Audio 2](https://arxiv.org/abs/2305.18474) | Temporal-enhanced text-to-audio generation | Time-frequency plots can later inspect temporal structure | Structured conditioning, model inference and temporal metrics |
| [Stable Audio Open](https://arxiv.org/abs/2407.14358) | Open-weights text-to-audio model | Generic loading and plotting will later inspect baseline outputs | Model download and inference; this demo's 24 kHz mono is not its model input/output specification |
| [DCASE 2023 Foley synthesis](https://dcase.community/challenge2023/task-foley-sound-synthesis-results) | Foley synthesis evaluation benchmark | General signal inspection only | Dataset protocol, FAD, listening tests, category fit and diversity evaluation |

The connections above are our interpretation of where basic signal tools can be useful; they do not imply that the notebook implements these papers. RMS is not a perceptual loudness measurement or a paper-specific loudness extractor. Mel-band count and FFT settings are analysis parameters, not generation conditions.

## Results and traceability

- [Our experiment notes](../../experiments/01_audio_basics/README.md)
- [Run metadata](../../experiments/01_audio_basics/run.json)
- [Numerical results](../../results/tables/01_audio_basics/summary.csv)
- [Figures](../../results/figures/01_audio_basics/)

**Published results:** T-FOLEY Table 1 and F-RAVE Tables 1–2 are extracted in their linked notes below. Other papers are compared for conditioning and availability; complete quantitative reviews remain pending. Our experiment metrics remain separate from author-reported values.

## Rule for the next notebook

For every notebook or reusable code module, record: exact source path, related paper/source, implemented component, status (preparation / partial implementation / inference / reproduction), missing components, run settings, measured results, output paths, limitations and source version. Separate author-reported results from results measured by this project.


## 02_feature_extraction.ipynb

**Status:** Feature-extraction preparation. [Notebook](../../notebooks/02_feature_extraction.ipynb) and [experiment](../../experiments/02_feature_extraction/README.md).

| Related work / source | Current implementation | Difference / remaining work |
|---|---|---|
| [Sketch2Sound, Section II-A](https://arxiv.org/html/2412.08550v2#S2.SS1) | Per-frame RMS and magnitude-weighted spectral centroid in Hz | RMS is unweighted rather than the paper's A-weighted loudness. No MIDI-like centroid scaling, CREPE probabilities, latent alignment, median-filter conditioning or generator is implemented. |
| [librosa official feature implementation](https://librosa.org/doc/0.11.0/_modules/librosa/feature/spectral.html) | Waveform RMS and STFT magnitude centroid | RMS uses a rectangular frame; centroid uses a Hann-window STFT. Same frame centers, different window weights. |

The -60 dBFS RMS mask is this demo's analysis choice, not the paper's evaluation mask. Published model results remain separate from the [measured CSV](../../results/tables/02_feature_extraction/summary.csv). Fixed gain ×0.5 reduces RMS by 6.0206 dB and preserves centroid on the same original-frame mask; this verifies feature behavior, not model disentanglement.


## 03_brightness_at_constant_rms.ipynb

**Status:** Controlled synthetic feature experiment. [Notebook](../../notebooks/03_brightness_at_constant_rms.ipynb); [experiment results](../../experiments/03_brightness_at_constant_rms/README.md).

The experiment increases a 2000 Hz component relative to a 200 Hz component, then rescales every signal to RMS 0.15. It checks whether magnitude-weighted centroid changes while global and frame RMS remain constant. This investigates centroid as a candidate descriptor; it does not implement a generator, perceptual brightness test or model disentanglement benchmark.

Relation to Sketch2Sound is limited to the centroid descriptor discussed in [our paper note](sketch2sound_controls.md). No paper results are reproduced. Constant RMS does not imply constant perceptual loudness or pitch.


## 04_real_audio_analysis.ipynb

Gerçek kayıt üzerinde ortak RMS/centroid analizi. `src/audio/real_sample.py` lisanslı InspectorJ Robin kaynağının librosa 22050 Hz mono OGG türevini doğrular ve FLOAT WAV'a çözer. `src/audio/features.py` RMS/centroid ve düşük enerji maskesini hesaplar. [Deney](../../experiments/04_real_audio/README.md) sonuçları gerçek analizdir; hiçbir paper generator'ı uygulanmaz. Sketch2Sound/F-RAVE descriptor ölçümlerine hazırlık; A-weighted veya makaleye özel normalize extractor değildir.

## 05_pitch_periodicity.ipynb

[pYIN](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html) F0 ve voicing probability; yalnız güvenilir voiced karelerde normalize ACF proxy'si. Sketch2Sound'un CREPE pitch/periodicity extractor'ı ile eşdeğer değildir. [Ölçümler](../../experiments/05_pitch_periodicity/README.md) ve 440 Hz/noise/silence diagnostics kaydedildi. F0 ground truth'u olmayan kuş kaydında accuracy iddiası yok.

## 06_tfoley_inference.ipynb

**Durum: resmî pretrained model inference; benchmark reprodüksiyonu değil.** `src/baselines/tfoley_inference.py`, [T-FOLEY](tfoley_baseline.md) model/sampler/SDE ve EMA checkpoint'ini kullanır; sınıf + RMS koşullama. Tek cihaz yerleştirme düzeltmesi, taşınabilir I/O ve checkpoint yükleme uygulanır. [Koşum ve ölçülen sonuçlar](../../experiments/baselines/tfoley/README.md) yazarın Table 1 sonuçlarından ayrıdır. Pitch, parlaklık, eğitim ve çoklu örnek kalite/kontrol benchmark'ı uygulanmadı.

## Güncel uygunluk kararı

[F-RAVE](frave_controls.md) text-free sürekli çoklu kontrol için yöntem önceliği; T-FOLEY hazır inference için baseline. [Karşılaştırma](literature_comparison.md) text/hybrid bağımlılıkları ve doğrulanmamış erişimi ayrı gösterir. Önceki eşleşmeler, notebook'ların makaleleri gerçeklemiş olduğunu ifade etmez.



## 07_tfoley_real_footstep.ipynb

T-FOLEY resmî pretrained inference + gerçek Footstep RMS referansı. `src/audio/footstep_sample.py` CC0 kamuya açık MP3 önizlemesini checksum doğrular, sabit 5–9 s crop/resample yapar. `src/audio/tfoley_comparison.py` aynı smoothing ile tek çift RMS MAE/korelasyon ve RMS peak zamanı tanılarını kaydeder. Peak zamanı onset değildir; bu timbre taklidi veya paper E-L1 benchmark reprodüksiyonu değildir. [Deney](../../experiments/baselines/tfoley/real_footstep/README.md).

[Referans-audio seçim notu](reference_audio_selection.md): AudioLDM waveform/CLAP yolu notebook 08 ilk CPU koşumunda çalıştırıldı; algısal başarı ölçülmedi. F-RAVE sayısal kontrol yöntemi incelemesi olarak kalır.


## 08_audioldm_reference_audio.ipynb

Resmî AudioLDM m-full pretrained audio-reference inference. CLAP audio koşulu; text prompt yok, empty unconditional branch var. `src/baselines/audioldm_reference_inference.py`: pinned source/checkpoint/tokenizer, safe mmap load, şartlı dropout 0, candidate 1, text-scoring guard, portatif reference I/O. [Koşum/sonuç](../../experiments/baselines/audioldm/README.md), [makale ilişkisi ve değişiklikler](audioldm_reference_baseline.md). Model mimarisi/yeniden eğitim, FAD veya bağımsız RMS/pitch/brightness kontrolü uygulanmadı.

Notebook 07'nin kullanıcı GPU çıktıları [ayrı provenance ile kaydedildi](../../experiments/baselines/tfoley/real_footstep/user_gpu_results.md); yerel WAV burada doğrulanmadı.
