# HierHypGen: Sheaf-Theoretic Diffusion for Structurally Consistent Image Generation

**Author:** Jorge A. Maure — DevIgnite LLC  
**Date:** February 2026  
**Hardware:** NVIDIA RTX 3060 12GB — all experiments on a single consumer GPU

---

## Abstract

HierHypGen is a novel generative architecture that uses **cellular sheaf theory** as the foundational mathematical framework for image generation. Unlike standard diffusion models that learn coherence from data, HierHypGen encodes structural consistency directly into the latent representation through restriction maps on graph edges and Sheaf Laplacian regularization.

To our knowledge, this is the **first application of cellular sheaves as a latent representation for generative image modeling**, and the **first sheaf-guided adapter for a foundation diffusion model**.

---

## Key Results

| Phase | Key Innovation | FID ↓ | Resolution |
|---|---|---|---|
| Phase 1 (8×8) | Fixed sheaf, synthetic data | 517 | 256×256 |
| Phase 1.8 (16×16) | Higher-resolution grid | 317 | 256×256 |
| Phase 3.5 | **Vector R + layer-wise prediction** | **275** | 256×256 |
| Phase 4 | Real data + CLIP conditioning | 260 | 256×256 |
| Phase 5 (optimized) | LPIPS decoder + extended training | 336 | 512×512 |
| U-Net Baseline 512 | Pixel-native baseline | 395 | 512×512 |
| **Phase 7 (FLUX+Sheaf)** | **DiT backbone + 1.3M sheaf adapter** | **—** | **512×512** |

At 512×512, HierHypGen **outperforms the pixel-native U-Net baseline by 58 FID points.**

---

## Core Contributions

**1. First sheaf latent representation for image generation.**  
Images are encoded into a sheaf over a spatial grid, with stalk features at nodes and restriction maps at edges.

**2. Sheaf Laplacian regularization.**  
Replaces KL divergence with a structural consistency penalty — the model does not need to *learn* that regions must be consistent, the Laplacian *enforces* it.

**3. Vector restriction maps with layer-wise topology prediction.**  
Each GNN layer independently predicts per-channel restriction maps. Layer-wise topology variance increases monotonically: `[0.14, 0.22, 0.38, 0.36, 0.56, 0.56]` — confirming coarse-to-fine structural differentiation.

**4. Sheaf adapters for foundation models.**  
A 1.3M-parameter Sheaf Latent Denoiser refines FLUX.2 Klein 4B's latent space — using only **0.03% of the base model's parameters**.

---

## Architecture

```
Input Image (B, 3, 512, 512)
    │
    ▼
NodeEncoder512 (5 Conv layers, frozen)
    │
    ▼
Normalize (per-channel latent stats)
    │
    ▼
Flow Matching (rectified flow interpolation)          FrozenCLIPEncoder
    │                                                      │
    └──────────────────────────────────────────────────────┘
                              │
                    SheafDenoiser (6 blocks)
                    ├── TextCrossAttention
                    ├── RestrictionMapPredictor → R_fine, R_coarse
                    ├── AdaLN (timestep + text conditioning)
                    ├── SheafConv_fine
                    ├── SheafConv_coarse
                    ├── GlobalPulse
                    └── MLP_node
                              │
                    Euler ODE Solver (50 steps)
                              │
                    NodeDecoder512 → (B, 3, 512, 512)
```

---

## Model Parameters

| Component | Params |
|---|---|
| SheafAE512 (Encoder + Decoder + R predictor) | ~1.53M |
| SheafDenoiser (6 blocks) | ~17.7M |
| FrozenCLIPTextEncoder | ~63M (frozen) |
| **Total trainable** | **~19.2M** |
| **Phase 7: SheafLatentDenoiser on FLUX Klein** | **~1.3M trainable** |

---

## Phase 7: FLUX Klein + Sheaf Adapter

The key insight of Phase 7: instead of building a complete generative model, apply the sheaf framework as a **lightweight structural refinement adapter** on top of a state-of-the-art foundation model.

- Base model: FLUX.2 Klein 4B (frozen)
- Adapter: 1.3M trainable parameters
- Training: 2000 augmented images, 200 epochs, 0.38 GB peak VRAM
- Result: Flow matching loss reduced from 2.28 → 1.55
- Adapter gates opened from 0.018 → 0.027 (quantifying sheaf contribution)

---

## Read the Paper

[HierHypGen_Paper.pdf](./HierHypGen_Paper.pdf)

---

## Citation

```
Maure, J. A. (2026). HierHypGen: Sheaf-Theoretic Diffusion for Structurally
Consistent Image Generation. DevIgnite LLC AI Research & Development.
https://github.com/Devignite25/AI-Research-Development
```

---

*© 2026 DevIgnite LLC — AI Research & Development. Licensed under CC BY 4.0.*
