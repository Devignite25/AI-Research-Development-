# HDAGI: Hierarchical Developmental Artificial General Intelligence

**Author:** Jorge A. Maure — DevIgnite LLC / The Wider Lens Research Initiative  
**Date:** December 2025 — March 2026  
**Hardware:** NVIDIA RTX 3060 12GB — full system on consumer hardware  
**Status:** Architecture validated ✅ | Phase 2 training in progress 🔄

---

## Overview

HDAGI is a developmental AI architecture that synthesizes four independent DevIgnite research programs into a single unified system. It grows in capability over time through five Piaget-inspired stages, uses uncertainty-triggered depth for compute-adaptive reasoning, encodes safety as structural constraints rather than learned penalties, and dreams in photorealistic imagined environments via the DIAMOND world model.

> *"Intelligence emerges from constrained growth, not maximized optimization."*

---

## The Synthesis

Each prior DevIgnite research program solved one critical problem — HDAGI unifies all four solutions:

| Prior Research | Problem Solved | How HDAGI Uses It |
|---|---|---|
| [FAA / Competence-Conservatism Paradox](../competence-conservatism-paradox/) | Learned safety suppresses capability | Structural domain-aware safety thresholds |
| [HTMA](../HTMA/) | Stateless attention is a hardware-era constraint | SSM (Mamba) backbone — persistent hierarchical memory |
| [HierHypGen](../HierHypGen/) | Statistical learning fails at high compression | Safety encoded structurally, not statistically |
| DIAMOND (Alonso et al., 2024) | Abstract imagination is unrealistic | Photorealistic diffusion-based dreaming |

---

## Architecture at a Glance

```
Input (Vision + Text)
        │
        ▼
HRM Backbone (SSM — no attention)
├── High-Level Planner (slow, strategic)
└── Low-Level Executor (fast, reactive)
    └── Uncertainty → triggers deeper recurrence
        │
        ▼
Cognitive Extensions (stage-gated)
├── Recursive Introspection
├── DIAMOND Dreaming (photorealistic)
├── Affective Morphing (mood-based LR)
└── Symbiotic Evolution (weight sharing)
        │
        ▼
Developmental System
├── ODE Growth Controller (dP/dt = r·P·(1-P/K))
├── Stage Manager (5 Piaget stages)
└── LoRA Spawner (reversible capacity growth)
        │
        ▼
Safety Layer (structural — not learned)
├── Domain-aware risk thresholds (FAA fix)
└── Productive Conflict (tolerates disagreement)
        │
        ▼
Output (Action + Value)
```

---

## Five Developmental Stages

| Stage | Capabilities | Status |
|---|---|---|
| SENSORIMOTOR | Basic action-outcome learning | ✅ Active |
| PRE_OPERATIONAL | + Dreaming + Affective Morphing | Pending |
| CONCRETE | + Recursive Introspection | Pending |
| FORMAL | + Symbiotic Evolution | Pending |
| METACOGNITIVE | All + Meta-learning | Pending |

---

## Hardware Profile

| Component | Parameters | VRAM |
|---|---|---|
| HDAGI core | 7.4M | ~0.7 GB |
| DIAMOND world model | 13.5M | ~2-3 GB |
| Total | **20.9M** | **~5-6 GB** |
| RTX 3060 available | — | 12 GB ✅ |

---

## Key Design Principles

**1. Structural safety over learned safety.**  
The Competence-Conservatism Paradox proved that learned safety critics suppress capability through correct operation. HDAGI uses domain-aware structural thresholds — the safety layer cannot be gamed by gradient pressure.

**2. Uncertainty-triggered depth.**  
The system recurs more deeply only when uncertain. Compute scales with difficulty, not with architecture size.

**3. Stage gating.**  
Powerful mechanisms (introspection, symbiotic evolution, meta-learning) unlock only when developmental prerequisites are met. No premature access.

**4. Additive growth signal.**  
signal = error + variance_bonus (additive, not multiplicative). Prevents runaway growth and growth starvation — validated through OLG research.

---

## Training Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Basic DIAMOND integration, Breakout | ✅ Complete |
| Phase 2 | Mixed-domain (Breakout + Pong), OLG growth | 🔄 In Progress |
| Phase 3 | 100-500 episode run, first stage transition | Planned |
| Phase 4 | Custom DIAMOND on MiniGrid | Future |

---

## Read the Paper

[HDAGI_Paper.pdf](./HDAGI_Paper.pdf)  
[HDAGI_Paper.md](./HDAGI_Paper.md)

---

## Citation

```
Maure, J. A. (2026). HDAGI: Hierarchical Developmental Artificial General
Intelligence — A Unified Architecture Synthesizing Four Independent Research
Programs. DevIgnite LLC / The Wider Lens Research Initiative.
https://github.com/Devignite25/AI-Research-Development
```

---

*© 2026 DevIgnite LLC — AI Research & Development. Licensed under CC BY 4.0.*
