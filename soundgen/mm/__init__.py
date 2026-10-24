"""Generator: an MMAudio-style flow-matching multimodal transformer (Sony / UIUC, CVPR 2025; code MIT).

We use MMAudio's network, VAE and flow-matching code as a library and train the weights ourselves:

    setup.py     clone MMAudio (pinned commit) + external weights (44.1 kHz mel-VAE, empty-string text feature)
    features.py  pack -> captions -> VAE latents + CLIP text tokens, cached once (memmaps)
    train.py     flow-matching training loop (text-only now; video features are "empty" until phase 2)
    generate.py  prompt (or mix of prompts) -> 3 variations + 5 blends between them -> 44.1 kHz audio

Architecture and reasons: docs/ARCHITECTURE.md.
"""
