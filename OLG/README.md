# OLG: Ontogenetic Layer Genesis

**Author:** Jorge A. Maure — DevIgnite LLC / The Wider Lens Research Initiative  
**Date:** December 2025  
**Hardware:** NVIDIA RTX 3060 12GB  
**Status:** ✅ Trained and validated — 0.95 reward, 69 minutes

---

## Overview

OLG is the **first paper** in the DevIgnite research program and the architectural origin of everything that follows. It proves that signal-driven developmental growth works on consumer hardware — a 133M parameter Mamba SSM backbone that spawns new layers when learning is difficult, guided by ODE dynamics and three-stage developmental gating.

Every subsequent DevIgnite architecture (FAA, HDAGI, Genesis) builds directly on the growth primitives first validated here.

---

## Results

| Metric | Value |
|---|---|
| Best reward | **0.95** |
| Training time | 69 minutes |
| Layers spawned | 4 (signal-based) |
| Parameters | 133M → 134M |
| Peak VRAM | 0.66 GB / 12.5 GB |
| Episodes | 2,429 |

---

## Core Insight

> Signal-based growth (4 layers, 0.95 reward) outperforms forced growth (9 layers, 0.61 reward).

More capacity is not better capacity. The right capacity at the right time is what matters.

---

## Architecture

```
Zygote Core (Mamba SSM)
        │
        ▼
Growth Controller
signal = smoothed_error + variance_bonus
if signal > 0.1 for 10 consecutive steps → SPAWN LAYER
        │
        ▼
Layer Spawning (LoRA adapters + FFN layers)
        │
        ▼
Stage Manager
SENSORIMOTOR → SYMBOLIC → METACOGNITIVE
(hybrid hard-coded + classifier gating)
```

---

## What This Established

- **Counter-based triggering** prevents noise spikes from causing spurious growth
- **Additive signal formula** (error + variance) is stable; multiplicative formulas are not
- **Stage gating** must be structural (hard-coded veto), not probabilistic
- **ODE logistic dynamics** provide natural growth ceilings without explicit cutoffs
- **Consumer hardware** is sufficient — 0.66 GB VRAM for a 133M parameter system

These five findings are directly inherited by every subsequent paper in the series.

---

## Experiment Log

| # | Steps | Layers | Reward | Notes |
|---|---|---|---|---|
| 001 | 5K | 0 | 0.86 | Baseline |
| 002 | 30K | 9 | 0.61 | Forced growth — too many |
| 003 | 50K | 3 | 0.94 | Signal-based — fixed |
| 004 | 200K | **4** | **0.95** | Optimal signal-based |

---

## Read the Paper

[OLG_Paper.pdf](./OLG_Paper.pdf)

---

## Citation

```
Maure, J. A. (2025). OLG: Ontogenetic Layer Genesis — A Bio-Inspired
Developmental AI Architecture. DevIgnite LLC / The Wider Lens Research
Initiative. https://github.com/Devignite25/AI-Research-Development
```

---

*© 2026 DevIgnite LLC. Licensed under CC BY 4.0.*
