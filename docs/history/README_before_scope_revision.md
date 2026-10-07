> Tarihsel README: güncel proje kapsamı [PROJECT_SCOPE.md](../PROJECT_SCOPE.md), aktif giriş [README](../../README.md). Aşağıdaki öncelikler tarihsel kararları yansıtır.

# Controllable Sound Generation Playground

## 09–10: Parametrik neural üretim

[09 doğrudan RMS kontrolü](../../notebooks/09_parametric_tfoley.ipynb) pretrained TFOLEY'yi sayısal koşul vektörüyle çalıştırır. [10 RMS + centroid CVAE](../../notebooks/10_parametric_spectral_cvae.ipynb) bu repo için eğitilmiş küçük neural spektrum modelini iki sayısal parametreyle çalıştırır. **10'un eğitimli küçük JSON ağırlıkları repoda; yeniden eğitim zorunlu değil.** Mevcut PyTorch/TFOLEY `.venv` kullanılır, AudioLDM kurulumu gerekmez.

Üç seed'li 12 CVAE üretiminde iki kontrol monoton değişti; diğer kontrol üzerindeki etki ayrıca ölçüldü. Tek kayıt ve random-phase rendering nedeniyle bu doğal/genel Foley modeli sonucu değildir. [Sonuç ve yeniden üretim](../../experiments/parametric/README.md), [model seçimi ve makale eşleşmesi](../../papers/notes/parametric_model_selection.md). Parametrik kontrol açısından TFOLEY enerji/zaman baseline, küçük CVAE çalışan iki-kontrol prototipi, F-RAVE ölçekleme için yöntem adayıdır.



## 08: Referans-audio baseline ve yerel GPU sonuçları

[Notebook 08](../../notebooks/08_audioldm_reference_audio.ipynb) AudioLDM m-full'ü kuş referansının audio embedding'iyle çalıştırır. Kullanıcı text prompt yok; empty unconditional branch modelde kalır. **İlk CPU üretimi tamamlandı**; düşük seviyeli çıktının algısal başarısı henüz ölçülmedi. [Koşum/kurulum](../../experiments/baselines/audioldm/README.md), [makale/kod eşleşmesi](../../papers/notes/audioldm_reference_baseline.md).

Model ayrı `.venv-audioldm` ortamında; kurulum notebook başında. İlk checkpoint yaklaşık 4.57 GB. Mevcut T-FOLEY/CUDA `.venv` korunur. Notebook referans, ortak gain çıktı ve ayrıca normalize edilmiş içerik dinlemesi sağlar.

[07 kullanıcı GPU sonuçları](../../experiments/baselines/tfoley/real_footstep/user_gpu_results.md) ayrı kaydedildi: 30.656 s, RMS Pearson 0.77258, 8/7 RMS peak. Sonuçlar kullanıcı tarafından sağlandı; WAV burada alınmadı. Bütün hedefleri karşılayan tek model henüz seçilmedi; T-FOLEY temporal, AudioLDM referans-içerik ve F-RAVE numeric-descriptor rolleri ayrı.

## Referans ses hedefi için güncelleme

Notebook 06 ilk dinlemede beklenen benzerliği sağlamadı. **T-FOLEY temporal baseline**, genel ses/timbre taklit modeli değildir. [Notebook 07](../../notebooks/07_tfoley_real_footstep.ipynb) gerçek Footstep referansını ve çıktıyı aynı gain ile dinletir; RMS/zaman tanılarını kaydeder. Kuş/çevresel referans içeriği için sıradaki aday **AudioLDM audio-to-audio**; metinsiz inference yolu kodda doğrulandı fakat ilk CPU kuş koşumu tamamlandı, algısal başarı ölçülmedi. [Güncel seçim](../../papers/notes/reference_audio_selection.md), [gerçek Footstep deneyi](../../experiments/baselines/tfoley/real_footstep/README.md).

## Önceki temporal / descriptor kararı — 2026-10-07

Ana hedef: **serbest metin gerektirmeyen, ölçülebilir ses kontrolü**. İlk hazır-model baseline **T-FOLEY** (sınıf + RMS/zaman). Sürekli çoklu öznitelik ve ayrıştırma için ilk yöntem adayı **F-RAVE**; uygun Foley checkpoint'i henüz doğrulanmadı. T-FOLEY pitch/parlaklık kontrolü sunmaz. Aşağıdaki eski aday listesi tarihsel kapsam tartışmasıdır; güncel öncelik [karşılaştırma ve seçim notundadır](../../papers/notes/literature_comparison.md).

| Notebook | Durum / sonuç kaydı |
|---|---|
| [01 Audio basics](../../notebooks/01_audio_basics.ipynb) | Sentetik sinyal analizi |
| [02 Feature extraction](../../notebooks/02_feature_extraction.ipynb) | RMS ve centroid; gain deneyi |
| [03 Brightness at constant RMS](../../notebooks/03_brightness_at_constant_rms.ipynb) | Sabit RMS ile centroid değişimi |
| [04 Real audio analysis](../../notebooks/04_real_audio_analysis.ipynb) | Lisanslı gerçek WAV; [sonuç](../../experiments/04_real_audio/README.md) |
| [05 Pitch / periodicity](../../notebooks/05_pitch_periodicity.ipynb) | pYIN, güvenilir kare maskesi ve ACF; [sonuç](../../experiments/05_pitch_periodicity/README.md) |
| [06 T-FOLEY inference](../../notebooks/06_tfoley_inference.ipynb) | Metinsiz pretrained inference; [koşum kaydı](../../experiments/baselines/tfoley/README.md) |

Paylaşılan özellik kodu `src/audio/`, model çalıştırıcısı `src/baselines/` altındadır. [Kod–çalışma eşleşmesi](../../papers/notes/code_literature_map.md) ve [karşılaştırma CSV](../../results/tables/literature_comparison.csv) günceldir.

Temel ortam: `pip install -r requirements.txt`. T-FOLEY için mevcut PyTorch/CUDA kurulumundan sonra `pip install -r requirements-tfoley.txt`. 04 ve 05 gerçek ses kaynağını checksum doğrulayarak indirir; 06 resmî kaynak ve checkpoint'i indirir. Büyük ağırlıklar, haricî kaynak checkout'u ve WAV dosyaları Git dışında; yeniden üretim kodu, kaynak/lisans bilgisi, tablo, grafik ve koşum metadata'sı Git içindedir.

Güncel parametrik deneyler 09–10 ve `experiments/parametric` altındadır. Sonraki kalite/genelleme eşiği çoklu kayıtlarla kayıt bazında ayrılmış eğitim ve değerlendirmedir.

## 1. Project Overview

This repository is a research playground for investigating **deep learning-based controllable sound generation**.

The long-term motivation is to develop sound generation methods that can eventually be integrated into applications such as cinematic sound design. However, **video-to-audio generation is currently outside the scope of this research**.

The current focus is on understanding how generative audio models can produce high-quality sound effects while allowing explicit and measurable control over the generated sound.

### Main Research Direction

**Controllable Sound Generation using explicit acoustic, perceptual, semantic, and temporal conditioning.**

---

## 2. Motivation

Many modern sound generation models rely primarily on natural-language prompts.

A conventional system can be represented as:

```text
Text Prompt
    ↓
Text Encoder
    ↓
Condition Representation
    ↓
Audio Generator
    ↓
Generated Audio
```

For example:

```text
"A loud metallic impact"
```

Text conditioning provides significant semantic flexibility, but it does not necessarily provide precise control over individual properties of the generated sound.

Terms such as:

- loud
- bright
- heavy
- sharp
- distant

must first be interpreted by a language model or text encoder.

For applications requiring precise sound manipulation, a more explicit conditioning representation may therefore be useful.

Instead of relying entirely on natural language, a sound could potentially be described using structured parameters:

```text
event       = impact
loudness    = 0.80
brightness  = 0.60
duration    = 1.20 s
```

The generator could then follow:

```text
Structured Parameters
        ↓
Condition Encoder
        ↓
Audio Generator
        ↓
Generated Audio
```

The central motivation of this project is therefore to investigate whether **explicit conditioning can provide more precise and measurable control over sound generation**.

---

# 3. Research Scope

### Included

- Sound-effect generation
- Environmental sound generation
- Foley sound generation
- Controllable audio generation
- Conditional diffusion models
- Structured conditioning
- Continuous acoustic conditioning
- Perceptual audio attributes
- Temporal sound control
- Multi-parameter conditioning
- Parameter disentanglement

### Currently Outside the Scope

- Video-to-audio generation
- Automatic video understanding
- Speech synthesis / Text-to-Speech
- Voice cloning
- Music generation

These areas may become relevant later, but they are not part of the current research problem.

---

# 4. Main Research Questions

The initial research question is:

> **Can explicit acoustic and perceptual conditioning provide more precise and measurable control over sound generation than conventional text-based conditioning?**

A more specific question is:

> **How accurately and independently can multiple acoustic or perceptual parameters be controlled in deep generative audio models?**

These questions may evolve as the literature review and experiments progress.

---

# 5. Conditioning Strategies

Three major conditioning strategies will initially be considered.

## 5.1 Text Conditioning

```text
Text
 ↓
Text Encoder
 ↓
Audio Generator
 ↓
Audio
```

Example:

```text
"A powerful metallic impact"
```

### Advantages

- Flexible
- Semantically expressive
- Easy for humans to use
- Widely supported by pretrained models

### Potential Limitations

- Ambiguous descriptions
- Limited fine-grained control
- Dependence on text encoder representations
- Difficult to quantitatively verify individual attributes

---

## 5.2 Parametric / Structured Conditioning

Instead of natural language, explicit parameters are provided.

Example:

```text
event       = impact
loudness    = 0.8
brightness  = 0.5
duration    = 1.4
```

Conceptually:

```text
Condition Vector
       ↓
Condition Encoder
       ↓
Audio Generator
       ↓
Audio
```

The condition vector may contain measurable or interpretable properties of the target sound.

---

## 5.3 Hybrid Conditioning

Text and explicit parameters may also be combined.

```text
Text ─────────┐
              ├──→ Audio Generator → Audio
Parameters ───┘
```

Example:

```text
Text:
"Metal impact"

Parameters:
loudness   = 0.8
brightness = 0.6
duration   = 1.2 s
```

This approach may preserve the semantic flexibility of text while providing more precise control over specific audio characteristics.

---

# 6. Candidate Sound Parameters

The exact parameter space has **not yet been finalized**.

Potential parameters currently include:

### Semantic

- Event class
- Material
- Sound category

### Temporal

- Duration
- Event onset
- Event offset
- Event timing
- Event frequency

### Acoustic

- Loudness / energy
- Pitch
- Spectral centroid
- Spectral bandwidth
- Attack
- Decay

### Perceptual

- Brightness
- Roughness
- Intensity

Additional parameters may later be investigated.

Parameters should not be added arbitrarily.

For every candidate parameter we should ask:

1. Is it objectively measurable?
2. Can it be extracted automatically from audio?
3. Can humans perceive changes in it?
4. Can a generative model reliably control it?
5. Can it be changed without strongly affecting other parameters?

---

# 7. Evaluation Framework

The initial evaluation framework will focus on three major dimensions:

```text
QUALITY
   +
CONTROLLABILITY
   +
DISENTANGLEMENT
```

## 7.1 Audio Quality

Question:

> Does the generated audio sound realistic and convincing?

Potential evaluation methods:

- Fréchet Audio Distance (FAD)
- Human evaluation
- Mean Opinion Score (MOS)
- Distribution-based audio metrics

The DCASE Foley Sound Synthesis Challenge will be investigated as a reference evaluation protocol.

---

## 7.2 Controllability

Question:

> Does the model actually follow the parameter we provide?

Example:

```text
Requested loudness:

0.2
0.4
0.6
0.8
```

We can then measure the actual loudness of each generated audio sample.

Ideally:

```text
Requested parameter ↑
        ↓
Measured audio attribute ↑
```

Correlation and similar statistical measurements can later be used to quantify this relationship.

---

## 7.3 Disentanglement

Question:

> Can one parameter be changed without unintentionally changing other properties?

For example:

```text
brightness:

0.2 → 0.4 → 0.6 → 0.8
```

Ideally:

```text
brightness ↑

while

loudness ≈ constant
pitch    ≈ constant
duration ≈ constant
```

This will be particularly important when evaluating **multi-parameter controllability**.

---

# 8. Initial Models and Research Papers

## 8.1 Sketch2Sound

**Sketch2Sound: Controllable Audio Generation via Time-Varying Signals and Sonic Imitations**

ICASSP 2025.

### Main Idea

Uses continuous time-varying conditioning signals including:

- Loudness
- Brightness
- Pitch

The work demonstrates how pretrained text-to-audio generators can be extended with explicit continuous controls.

Paper:

https://arxiv.org/abs/2412.08550

Project:

https://research.adobe.com/publication/sketch2sound-controllable-audio-generation-via-time-varying-signals-and-sonic-imitations/

### Relevance to This Project

**Very High**

It is one of the closest existing approaches to the type of explicit parameter control investigated in this playground.

---

## 8.2 Audio ControlNet

**Audio ControlNet for Fine-Grained Audio Generation and Editing**

### Controls

- Loudness
- Pitch
- Sound-event timing

The method introduces additional control mechanisms around a pretrained audio-generation backbone.

Paper:

https://arxiv.org/abs/2602.04680

GitHub:

https://github.com/juhayna-zh/AudioControlNet

### Relevance to This Project

**Very High**

The implementation and pretrained components make this an important candidate for experimental benchmarking.

---

## 8.3 T-FOLEY

**T-FOLEY: A Controllable Waveform-Domain Diffusion Model for Temporal-Event-Guided Foley Sound Synthesis**

ICASSP 2024.

### Conditioning

```text
Sound Class
+
Temporal Event Information
```

T-FOLEY is particularly relevant because controllable sound generation can be performed without relying on free-form natural-language prompts.

Paper:

https://arxiv.org/abs/2401.09294

GitHub:

https://github.com/YoonjinXD/T-FOLEY

### Relevance to This Project

**High**

It can potentially serve as a **non-NLP baseline**.

---

## 8.4 Audio Generation with Multiple Conditional Diffusion Model

AAAI 2024.

### Conditioning

- Text
- Timestamp
- Pitch contours
- Energy contours

An important architectural concept is the use of a pretrained generator together with additional conditioning modules rather than training the entire generator from scratch.

Paper:

https://doi.org/10.1609/aaai.v38i16.29773

### Relevance to This Project

**High**

Useful for studying multi-condition architectures.

---

## 8.5 PicoAudio

PicoAudio investigates precise temporal control over generated audio events.

### Conditioning

```text
Event
+
Timing
+
Occurrence Information
```

GitHub:

https://github.com/zeyuxie29/PicoAudio

### Relevance to This Project

**Medium / High**

Useful for studying structured event conditioning and temporal controllability.

---

## 8.6 Make-An-Audio 2

**Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation**

The work investigates structured representations of audio events and their temporal relationships.

Paper:

https://arxiv.org/abs/2305.18474

GitHub:

https://github.com/bytedance/Make-An-Audio-2

### Relevance to This Project

**Medium**

Useful for understanding the transition from free-form text toward structured event representations.

---

## 8.7 Stable Audio Open

Stable Audio Open is an open generative audio model that can potentially serve as a baseline or pretrained backbone.

Simplified architecture:

```text
Text
 ↓
Text Encoder
 ↓
Diffusion Transformer
 ↓
Latent Audio
 ↓
Audio Decoder
 ↓
Generated Audio
```

Paper:

https://arxiv.org/abs/2407.14358

GitHub Demo:

https://github.com/Stability-AI/stable-audio-open-demo

### Possible Future Direction

```text
Stable Audio Open
        +
Parameter / Control Adapter
        ↓
Controllable Sound Generator
```

### Relevance to This Project

**High as a baseline/backbone**

---

# 9. Benchmark Reference

## DCASE 2023 — Foley Sound Synthesis

DCASE provides an established benchmark for generative Foley audio.

Example categories include:

- Dog Bark
- Footstep
- Gun Shot
- Keyboard
- Moving Motor Vehicle
- Rain
- Sneeze / Cough

The evaluation considers concepts such as:

- Fréchet Audio Distance
- Audio quality
- Category fit
- Diversity

Reference:

https://dcase.community/challenge2023/task-foley-sound-synthesis-results

The DCASE methodology may later be adapted when designing our own benchmark.

---

# 10. Additional Resources

## Awesome Audio Generation

A collection of audio-generation papers and implementations:

https://github.com/xiquan-li/Awesome-Audio-Generation

Of particular interest:

**Text-to-Audio with Fine-Grained Control**

This repository can be used to monitor new controllable audio-generation research.

---

## Implementation Resources

Implementation-focused resources can also be stored in the playground.

These should **not be considered primary academic references**.

Example:

**Tiny Audio Diffusion**

https://medium.com/data-science/tiny-audio-diffusion-ddc19e90af9b

Useful for understanding relatively lightweight diffusion-based audio generation.

---

# 11. Current Experimental Selection

1. **T-FOLEY:** executable prompt-free class + temporal RMS baseline.
2. **F-RAVE:** primary continuous multiattribute method candidate; domain-appropriate checkpoint remains unverified.
3. **Sketch2Sound / Audio ControlNet:** descriptor and hybrid-control references; text-free semantic capability must be demonstrated before promotion.
4. Text-conditioned models remain secondary comparators.

Full evidence and limitations: [literature comparison](../../papers/notes/literature_comparison.md).

---

# 12. Research Roadmap

## Phase 1 — Literature Review

Study controllable sound-generation literature.

For each paper, investigate:

- Problem
- Architecture
- Conditioning representation
- Dataset
- Loss functions
- Evaluation metrics
- Results
- Limitations
- Pretrained model availability

**Deliverable:** Literature comparison table.

---

## Phase 2 — Parameter Investigation

Determine which sound properties are suitable as explicit conditioning variables.

Initial candidates:

```text
loudness
energy
pitch
brightness
spectral centroid
duration
attack
decay
roughness
temporal events
```

**Deliverable:** Candidate parameter space.

---

## Phase 3 — Playground Implementation

Implement common utilities for:

```text
audio loading
resampling
waveform visualization
spectrogram analysis
feature extraction
model inference
audio generation
```

---

## Phase 4 — Baseline Experiments

Run selected pretrained models.

Generate controlled sets of audio samples.

Example:

```text
Parameter: Loudness

0.1
0.2
0.3
...
1.0
```

Measure whether the generated audio follows the requested control.

---

## Phase 5 — Benchmark

Evaluate models according to:

```text
Audio Quality
Controllability
Disentanglement
Diversity
Computational Cost
```

---

## Phase 6 — Gap Analysis

Compare experimental results with claims made in the literature.

Look for:

- Poorly controlled parameters
- Coupled parameters
- Unstable controls
- Limitations of text conditioning
- Limitations of existing control architectures

The goal is to identify a **specific research gap** before designing a new model.

---

## Phase 7 — Method Development

Only after identifying a clear limitation should a new method be proposed.

Potential directions include:

- Multi-parameter conditioning
- Continuous control adapters
- Parameter-specific encoders
- Disentanglement objectives
- Attribute consistency losses
- New conditioning representations

A future training objective may conceptually combine:

```text
Total Loss
    =
Generation Loss
    +
Control Loss
    +
Disentanglement Loss
```

The exact formulation will depend on experimental findings.

---

# 13. Repository Structure

The current repository follows the local `sound-gen` directory layout.

| Path | Current contents / purpose |
|---|---|
| `notebooks/` | `01_audio_basics.ipynb` and `02_feature_extraction.ipynb`: audio representations and frame features |
| `src/` | Reusable audio features and T-FOLEY inference adapter |
| `experiments/01_audio_basics/` | Experiment notes and run metadata |
| `papers/controllable_generation/` | Reserved for controllable-generation references |
| `papers/evaluation/` | Reserved for evaluation references |
| `papers/notes/` | Code–literature mapping and paper-note template |
| `data/raw/` | Local input recordings; ignored by Git |
| `data/processed/` | Local processed recordings; ignored by Git |
| `data/generated/` | Reproducible demo audio; ignored by Git |
| `results/audio/` | Local output audio; ignored by Git |
| `results/figures/01_audio_basics/` | Exported waveform, spectrogram and mel-spectrogram figures |
| `results/tables/01_audio_basics/` | Measured audio basics results |
| `.gitignore` | Excludes environments, audio and model weights; keeps folder placeholders |
| `requirements.txt` | Basic notebook dependencies; CUDA/PyTorch setup remains separate |

Empty directories are preserved with `.gitkeep`. Notebooks 01–08 and baseline/analysis result subdirectories are available; current paths are listed above.

---

# 14. Paper Notes

Each important paper should eventually have its own note.

Recommended template:

```text
Paper:
Authors:
Year:
Conference / Journal:

Problem:

Main Contribution:

Architecture:

Conditioning:

Dataset:

Loss Functions:

Evaluation Metrics:

Main Results:

Limitations:

Code Available:
Yes / No

Pretrained Model Available:
Yes / No

Relevance to Our Research:

Ideas / Questions:
```

This will make it easier to convert the playground into a structured literature review later.

---

# 15. Current Status

Completed: synthetic audio basics, frame RMS/centroid, constant-RMS centroid experiment, licensed real-audio analysis, masked pitch/periodicity diagnostics, literature comparison and selection, portable T-FOLEY inference runner.

Our first model run and actual metrics are recorded in [experiments/baselines/tfoley](../../experiments/baselines/tfoley/README.md). Dataset, multiattribute conditioning space, benchmark protocol and a new architecture are not finalized.

---

# 16. Immediate Next Steps

- T-FOLEY RMS gain/onset sweeps with fixed class/seed, then multiple seeds.
- Target/output error, temporal alignment and unintended descriptor changes.
- F-RAVE domain, checkpoint and reproducible training-cost audit.
- Quality/category listening evaluation and a matched dataset protocol.

Method development follows evidence of a specific control failure.

---

# 17. Run and Record the First Experiment

Open [`notebooks/01_audio_basics.ipynb`](../../notebooks/01_audio_basics.ipynb) in VS Code and select the project's Python 3.11 `.venv` kernel. The notebook works from the repository root or from `notebooks/`. Run all cells. With `AUDIO_PATH = None`, it creates a reproducible synthetic WAV locally without downloading data.

Outputs are saved automatically to `results/figures/01_audio_basics/`, `results/tables/01_audio_basics/` and `experiments/01_audio_basics/run.json`. Re-running replaces those outputs; preserve a copy before comparing different inputs.

- [Experiment and measured results](../../experiments/01_audio_basics/README.md)
- [Code–literature mapping](../../papers/notes/code_literature_map.md)
- [Paper-note template](../../papers/notes/paper_note_template.md)

The committed results were measured in the execution environment recorded in `run.json`. They are a synthetic signal-processing demonstration, not generated-model benchmarks or reproductions of paper results. The local Windows/Python 3.11 run is still to be confirmed.

For each future experiment, record the source notebook, related paper, implementation status, settings, results and limitations. Keep published paper results separate from our measured results.



## Frame Feature Extraction

[`notebooks/02_feature_extraction.ipynb`](../../notebooks/02_feature_extraction.ipynb) computes per-frame unweighted RMS, RMS level and spectral centroid, plots the curves, and checks their response to a fixed gain change. It runs independently with a synthetic demo or a user-supplied WAV.

- [Experiment settings and results](../../experiments/02_feature_extraction/README.md)
- [Frame values](../../results/tables/02_feature_extraction/frame_features.csv)
- [Feature curves](../../results/figures/02_feature_extraction/feature_curves.svg)

This is shared feature-extraction infrastructure, not a Sketch2Sound implementation. Pitch extraction is implemented separately in notebook 05. Outputs are overwritten on rerun; preserve earlier runs separately when comparing inputs.

