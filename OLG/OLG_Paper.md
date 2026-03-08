# OLG: Ontogenetic Layer Genesis — A Bio-Inspired Developmental AI Architecture

**Jorge A. Maure**  
DevIgnite LLC — AI Research & Development  
The Wider Lens Research Initiative  
Jacksonville, Florida, USA  
December 2025

---

## Abstract

We present **OLG (Ontogenetic Layer Genesis)**, a bio-inspired developmental AI architecture that grows from a minimal "zygote" network through staged phases, driven by intrinsic signals and ODE dynamics. OLG is the first system in the DevIgnite research program and provides the foundational growth mechanics that all subsequent architectures build upon.

Starting from a 133M parameter Mamba SSM backbone, OLG spawns new layers based on a learning difficulty signal (error + variance) using counter-based triggering. A three-stage developmental progression — SENSORIMOTOR → SYMBOLIC → METACOGNITIVE — gates capability unlocks. Anti-catastrophic-forgetting mechanisms (EWC + rehearsal + peak-based unfreezing) preserve prior knowledge during growth.

Empirical results on a six-environment MiniGrid curriculum demonstrate: best reward of 0.95, 4 signal-based layers spawned, 69 minutes training time, and 0.66 GB peak VRAM on RTX 3060 12GB. These results establish counter-based growth triggering as a reliable primitive and provide the empirical foundation for all subsequent DevIgnite research programs.

**Keywords:** developmental AI, neural architecture growth, ODE dynamics, curriculum learning, catastrophic forgetting, bio-inspired AI, Mamba SSM

---

## 1. Introduction

### 1.1 The Problem With Fixed-Capacity Architectures

Modern AI systems are initialized with a fixed number of parameters and trained until convergence. This approach has a fundamental mismatch with biological intelligence: brains do not start large and compress — they start minimal and grow in response to need.

Fixed-capacity architectures face two compounding problems. Overparameterization at initialization wastes compute on capacity that may never be needed. Underparameterization creates a hard ceiling that cannot be exceeded without retraining from scratch. Neither resembles how biological cognition develops.

### 1.2 Ontogenesis as an Architectural Principle

Ontogenesis — the developmental process by which a biological organism grows from a single cell to a complete system — provides the organizing principle for OLG. The key insight: **growth should be driven by learning difficulty, not by architectural decisions made before training begins.**

When a biological organism encounters a task beyond its current capability, it grows new capacity. When capacity is sufficient, growth stops. This is not a metaphor — it is a specific algorithmic principle: measure learning difficulty, trigger growth when difficulty exceeds current capacity, gate powerful capabilities until prerequisites are met.

### 1.3 Contributions

1. **Signal-based layer spawning.** New layers spawn when a learning difficulty signal (smoothed error + variance) exceeds a threshold for a sustained number of steps — counter-based triggering that prevents spurious growth from noise spikes.
2. **ODE-governed growth dynamics.** Layer population follows logistic growth: dP/dt = r × P × (1 - P/K), preventing runaway growth while guaranteeing eventual capacity saturation.
3. **Three-stage developmental gating.** SENSORIMOTOR → SYMBOLIC → METACOGNITIVE stages gate capability unlocks with hybrid hard-coded + classifier gating.
4. **Anti-catastrophic-forgetting suite.** EWC + experience rehearsal + peak-based weight unfreezing preserves prior performance during growth.
5. **Consumer hardware validation.** Full system at 133M parameters, 0.66 GB VRAM, RTX 3060 12GB.

---

## 2. Architecture

### 2.1 Overview

```
Zygote Core (Mamba SSM backbone)
        │
        ▼
Growth Controller (signal = error + variance)
        │
        ▼
Layer Spawning (LoRA vision, FFN)
        │
        ▼
Stage Manager (SENSORIMOTOR → SYMBOLIC → METACOGNITIVE)
```

### 2.2 Zygote Core

The minimal starting architecture is a Mamba SSM backbone — chosen for O(n) context handling, stable gradients, and efficient temporal modeling. The "zygote" metaphor is precise: the network starts minimal and all subsequent capacity is grown, not initialized.

Starting parameters: ~133M. This fits entirely within RTX 3060 12GB with significant headroom for growth.

### 2.3 Growth Controller

The growth signal combines two components:

```
signal = smoothed_error + variance_bonus
```

**Smoothed error** tracks the exponential moving average of prediction error — persistent difficulty, not noise. **Variance bonus** captures instability in the learning signal — the system is encountering novel states it cannot yet model.

Counter-based triggering:
```
if signal > threshold (0.1):
    steps_above_threshold += 1
else:
    steps_above_threshold = max(0, steps_above_threshold - 1)

if steps_above_threshold >= trigger_steps (10):
    SPAWN_LAYER()
    cooldown = 100 steps
```

The counter prevents noise spikes from triggering spurious growth. The decrement on sub-threshold steps (rather than reset to zero) prevents the counter from accumulating across separated difficulty spikes.

### 2.4 ODE Growth Dynamics

Layer population follows logistic growth:

```
dP/dt = r × P × (1 - P/K)
```

Where P = current parameters, K = maximum parameters, r = growth rate (0.01). This provides: rapid early growth when P << K, natural deceleration as P approaches K, and a hard ceiling without explicit cutoffs.

### 2.5 Layer Spawning

Two layer types spawn on growth trigger:

**LoRA vision adapters** — low-rank (rank=16, alpha=32) adapters applied to the vision pathway. Low-rank, reversible, cheap — correct growth primitive for perceptual capacity.

**FFN layers** — standard feedforward expansion for reasoning capacity. Inserted at the growth point in the network depth.

### 2.6 Stage Manager

Three developmental stages with hybrid gating (hard-coded prerequisites + learned classifier suggestion):

| Stage | Capabilities | Transition Gate |
|---|---|---|
| SENSORIMOTOR | Basic action-outcome, perception | Baseline performance threshold |
| SYMBOLIC | + Abstract representation, planning | Min steps + no regression |
| METACOGNITIVE | + Self-monitoring, meta-learning | Shadow evaluation pass |

Hard-coded gates veto premature transitions regardless of classifier suggestion. This is the key lesson carried forward into all subsequent architectures: **gate powerful capabilities structurally, not probabilistically.**

### 2.7 Anti-Catastrophic-Forgetting

Three mechanisms work together to preserve prior knowledge during growth:

**Elastic Weight Consolidation (EWC)** — penalizes changes to weights that were important for prior tasks, weighted by Fisher information. Prevents growth from overwriting learned representations.

**Experience Rehearsal** — a replay buffer of high-importance experiences from prior stages is mixed into each training batch. Importance = TD error × recency.

**Peak-Based Unfreezing** — weights are frozen after a performance peak is reached in each environment. They unfreeze only if performance drops below a threshold, allowing targeted repair without disrupting good weights.

---

## 3. Experiments

### 3.1 Setup

**Hardware:** NVIDIA RTX 3060 12GB  
**Curriculum:** 6 MiniGrid environments, increasing difficulty  
```
Empty-5x5 → FourRooms → DoorKey-5x5 → DoorKey-6x6 → DoorKey-8x8 → Mixed
```
**Baseline:** Fixed-capacity Mamba SSM, same initial parameters, no growth

### 3.2 Results Summary

| Metric | Value |
|---|---|
| Training time | 69 minutes |
| Layers spawned | 4 (signal-based) |
| Parameters: start → end | 133M → 134M |
| Peak GPU memory | 0.66 GB / 12.5 GB |
| Best reward | 0.95 |
| Total episodes | 2,429 |

### 3.3 Experiment Progression

| Experiment | Steps | Layers Spawned | Best Reward | Notes |
|---|---|---|---|---|
| 001 | 5K | 0 | 0.86 | Baseline — no growth |
| 002 | 30K | 9 | 0.61 | Forced growth — too many layers, performance drop |
| 003 | 50K | 3 | 0.94 | Signal-based — fixed threshold |
| 004 | 200K | 4 | **0.95** | Long training — optimal signal-based growth |

### 3.4 Key Finding: Signal-Based vs Forced Growth

Experiment 002 (forced growth at fixed intervals) spawned 9 layers and achieved only 0.61 reward — worse than the baseline. Experiment 004 (signal-based growth) spawned only 4 layers and achieved 0.95 reward.

This result establishes the core OLG principle empirically: **growth driven by learning difficulty outperforms growth driven by schedule.** More capacity is not better capacity. The right capacity at the right time is what matters.

### 3.5 Counter-Based Triggering Validation

Comparing pure threshold triggering (growth fires any time signal > 0.1) versus counter-based triggering (growth fires only after 10 consecutive steps above threshold) showed that pure threshold triggering caused 3× more spurious growth events and 12% lower final reward. The counter is essential — noise spikes in the learning signal are frequent and must not trigger architectural changes.

---

## 4. Discussion

### 4.1 The Additive Signal Formula

The growth signal uses addition (error + variance), not multiplication (error × variance). This is not arbitrary. Multiplicative signals produce extreme values when both components are high, causing either runaway growth or complete stagnation depending on initialization scale. The additive formula produces stable, interpretable signals across training phases.

This finding was directly carried forward into HDAGI and Genesis, where the same formula is used.

### 4.2 Staged Gating as a Safety Mechanism

The three-stage developmental gating is not primarily a performance optimization — it is a safety mechanism. METACOGNITIVE capabilities (self-monitoring, meta-learning) are powerful enough to destabilize prior learning if unlocked prematurely. Hard-coded prerequisites ensure these capabilities are only accessible when the system has demonstrated sufficient stability.

This principle was formalized in the Competence-Conservatism Paradox paper and carried into HDAGI and Genesis as the core architectural safety primitive.

### 4.3 Limitations

**Scale.** 133M → 134M growth is minimal. The ODE ceiling was not approached. Future work requires longer training runs and harder curriculum to stress-test the growth ceiling.

**Single modality.** OLG operates on MiniGrid observations only. Multimodal fusion is introduced in Genesis.

**No world model.** Imagination-based training (introduced via DIAMOND in HDAGI) is absent. OLG learns entirely from real environment interaction.

---

## 5. Conclusion

OLG establishes the foundational growth primitives for the DevIgnite research program: counter-based signal triggering, ODE-governed dynamics, staged developmental gating, and anti-forgetting mechanisms. The empirical result — 0.95 reward on a six-environment curriculum at 0.66 GB VRAM — validates these primitives on consumer hardware.

Every subsequent DevIgnite architecture (FAA, HDAGI, Genesis) builds directly on the OLG growth system. The counter-based trigger, the additive signal formula, the hybrid stage gating — these originated here.

---

**Code:** https://github.com/Devignite25/AI-Research-Development  
**Acknowledgments.** OLG was the first experiment in the DevIgnite research program, conducted in December 2025.

---

## References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
2. Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks (EWC). PNAS.
3. Chevalier-Boisvert, M., et al. (2023). MiniGrid: Minimalistic Gridworld Environments for Gymnasium. arXiv:2306.13831.
4. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
5. Maure, J. A. (2026). HDAGI: Hierarchical Developmental AGI. DevIgnite LLC. [This repository]
6. Maure, J. A. (2026). Genesis Architecture. DevIgnite LLC. [This repository]

---

*© 2026 DevIgnite LLC — AI Research & Development / The Wider Lens Research Initiative. Licensed under CC BY 4.0.*
