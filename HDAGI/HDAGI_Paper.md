# HDAGI: Hierarchical Developmental Artificial General Intelligence

**A Unified Architecture Synthesizing Four Independent Research Programs**

**Jorge A. Maure**  
DevIgnite LLC — AI Research & Development  
The Wider Lens Research Initiative  
Jacksonville, Florida, USA  
December 2025 — March 2026

---

## Abstract

We present **HDAGI (Hierarchical Developmental Artificial General Intelligence)**, a unified architecture that synthesizes four independent research programs conducted at DevIgnite LLC between December 2025 and March 2026. HDAGI integrates: (1) the FAA productive conflict resolution mechanism that solves the Competence-Conservatism Paradox; (2) the HTMA principle of continuous state evolution over persistent memory structures rather than stateless attention; (3) the HierHypGen insight that structural inductive bias outperforms statistical learning at high compression; and (4) the DIAMOND world model for realistic imagination-based training.

The result is a developmental AI architecture that grows in capability over time through five Piaget-inspired stages, uses uncertainty-triggered depth for compute-adaptive reasoning, encodes safety as a structural constraint rather than a learned penalty, and dreams in realistic imagined environments using a diffusion world model. HDAGI runs on consumer hardware (NVIDIA RTX 3060, 12GB VRAM) at 7.4M parameters with the DIAMOND integration bringing total parameters to 20.9M.

Architecture validation confirms HDAGI is internally consistent, avoids common AGI-architecture failure modes, and is implementable without institutional resources. Phase 2 training is underway.

**Keywords:** developmental AI, hierarchical recurrence, state space models, safety alignment, world models, growth architectures, AGI architecture

---

## 1. Introduction

### 1.1 The Problem With Existing Architectures

Modern AI systems are built around a fundamental assumption: scale solves everything. Transformers with more parameters, more data, and more compute eventually learn whatever structure is needed. This assumption has produced remarkable results — but it carries three deep problems.

**Structural blindness.** Systems that learn coherence statistically rather than encoding it mathematically remain fragile at distribution boundaries. HierHypGen demonstrated this directly: a sheaf-theoretic architecture with explicit structural constraints outperformed a pixel-native U-Net by 58 FID points at 512×512 resolution — not because it had more parameters, but because it encoded the right inductive bias.

**Safety suppression.** Systems with learned safety critics systematically suppress beneficial risk-taking once those critics become competent. The Competence-Conservatism Paradox, discovered during FAA research, showed that a trained F-Module reduced H-Module engagement from 55% to 11.6% — not through malfunction, but through correct operation. Standard RL cannot fix this. The Variance Asymmetry Problem guarantees that policy-gradient methods will always favor the low-variance conservative signal.

**Stateless amnesia.** Transformer architectures are fundamentally stateless between contexts. The HTMA proposal identified this as a hardware-era constraint masquerading as a design principle: attention mechanisms were optimal for GPU clusters circa 2017, not for intelligence itself. Unified memory systems remove the constraint. SSM-based architectures provide the alternative — infinite context in O(n) time.

### 1.2 The Synthesis Opportunity

Four independent research programs, each addressing one of these problems, produced converging insights:

| Research | Problem Addressed | Key Finding |
|---|---|---|
| FAA | Safety suppression | Competence-Conservatism Paradox + domain-aware fix |
| HTMA | Stateless amnesia | Continuous geometric state evolution over persistent memory |
| HierHypGen | Structural blindness | Sheaf-theoretic structural inductive bias |
| HDAGI + DIAMOND | All three | Unified developmental architecture with imagined training |

HDAGI is the architecture that emerges when these four threads are woven together.

### 1.3 Contributions

1. **Unified synthesis** of four independent research programs into a single coherent architecture.
2. **Developmental stage gating** inspired by Piaget's cognitive development theory — capabilities unlock sequentially, preventing premature access to powerful mechanisms.
3. **Uncertainty-triggered depth** — the system recurs more deeply only when uncertain, providing compute-adaptive "thinking harder."
4. **Structural safety** — safety is encoded as domain-aware thresholds (the FAA fix), not as a learned penalty subject to the Variance Asymmetry Problem.
5. **DIAMOND integration** — realistic dreaming via diffusion world model, replacing abstract noise imagination with photorealistic imagined rollouts.
6. **Consumer hardware implementation** — full system runs on RTX 3060 12GB at 20.9M total parameters.

---

## 2. Background and Prior Work

### 2.1 The Competence-Conservatism Paradox (FAA Research)

The FAA research program discovered that in a triad of H-Module (risk-taker), L-Module (safety net), and F-Module (safety critic), training the critic to competently detect conflict systematically suppresses the risk-taking module. The mechanism is the Variance Asymmetry Problem: because Var(R_H) >> Var(R_L), policy-gradient methods always converge to the stable low-variance L policy.

The validated fix — asymmetric absolute-confidence thresholds — is directly incorporated into HDAGI's Productive Conflict system. The key insight applied here: **safety must be structurally encoded, not learned**.

### 2.2 Continuous State Evolution (HTMA Research)

The HTMA proposal argued that transformer attention is optimal for GPU cluster computing circa 2017, not for intelligence. Unified memory removes the hardware constraint that made attention necessary. The architectural alternative: continuous geometric state evolution across persistent hierarchical memory manifolds.

HDAGI instantiates this principle using State Space Models (Mamba/S4) rather than attention. SSMs provide: infinite context in O(n) time, stable gradients, capture of rhythmic temporal patterns, and no quadratic memory scaling. The HRM backbone's bidirectional recurrence between high-level planner and low-level executor directly realizes the HTMA hierarchical memory manifold concept.

### 2.3 Structural Inductive Bias (HierHypGen Research)

HierHypGen demonstrated that encoding structural consistency mathematically (via Sheaf Laplacian regularization) outperforms learning it statistically at high compression ratios. The principle generalizes beyond image generation: whenever a system must operate in a compressed latent space, structural inductive bias provides stronger guarantees than statistical learning alone.

HDAGI applies this principle to its safety layer. Rather than learning safety from reward signals (which suffers from the Variance Asymmetry Problem), safety is encoded as structural domain-aware thresholds — the architectural analogue of Sheaf Laplacian regularization.

### 2.4 DIAMOND World Model

DIAMOND (DIffusion As a Model Of eNvironment Dreams) provides photorealistic imagination rollouts via a diffusion-based world model. Integrated into HDAGI as the DiamondDreamer, it replaces abstract noise-based imagination with realistic frame prediction. At Phase 2 training, HDAGI + DIAMOND totals 20.9M parameters with peak VRAM usage of 0.26 GB on RTX 3060 — well within consumer hardware constraints.

---

## 3. Architecture

### 3.1 Overview

```
INPUT STREAMS (Vision + Text)
        │
        ▼
MULTIMODAL EMBEDDER
        │
        ├──────────────────────────────┐
        ▼                              ▼
HIGH-LEVEL PLANNER            LOW-LEVEL EXECUTOR
(Slow, Strategic)    ◄────►   (Fast, Reactive)
SSM backbone                  SSM backbone
Goals ──────────────────────► 
◄─────────────────────────── Feedback
◄─────────────────────────── Uncertainty Signal
        │
        ▼
COGNITIVE EXTENSIONS (stage-gated)
├── Recursive Introspection (RIC)
├── Nested Dreaming via DIAMOND (NVDH)
├── Affective Morphing
└── Symbiotic Evolution
        │
        ▼
DEVELOPMENTAL SYSTEM
├── ODE Growth Controller
├── Stage Manager (5 Piaget stages)
└── LoRA Spawner
        │
        ▼
SAFETY LAYER
├── Safety Priors (structural thresholds)
└── Productive Conflict (FAA-derived)
        │
        ▼
OUTPUT HEADS
├── Action Head (policy)
└── Value Head (critic)
```

### 3.2 HRM Backbone

The Hierarchical Recurrent Model separates cognition into two timescales:

| Level | Purpose | Timescale | Layers |
|---|---|---|---|
| High | Strategic planning, goals | Slow | 3-4 |
| Low | Action execution, reactions | Fast | 4-6 |

Bidirectional communication: High sends goals to Low; Low sends feedback and uncertainty signals to High. When uncertainty exceeds a threshold, depth increases — the system recurs more deeply, implementing compute-adaptive "thinking harder" without fixed computational cost.

Each layer uses Mamba/S4 SSM:
- d_state: 64 (SSM hidden dimension)
- d_conv: 4 (local convolution width)
- expand: 2 (expansion factor)

This directly realizes the HTMA principle: persistent hierarchical memory manifolds with continuous state evolution, no attention, no quadratic scaling.

### 3.3 Cognitive Extensions

All cognitive extensions are **stage-gated** — they unlock only when developmental prerequisites are met. This is the key architectural decision that prevents premature access to powerful mechanisms.

**Recursive Introspection (RIC).** The system builds a self-model, predicts its own outputs, detects mismatches, and applies correction gradients. Parameters: max_depth=3, rewind_threshold=0.2, gradient_scale=0.1. Unlocks at CONCRETE stage. Risk: gradient injection can destabilize during symbiosis — mitigation via IntrospectionEnergyLimiter (capped correction energy per step).

**Nested Dreaming via DIAMOND (NVDH).** Memory consolidation and counterfactual imagination using DIAMOND's diffusion world model. Micro-dreams (bottom-up VAE replay) and macro-dreams (top-down goal-directed imagination). Maximum nesting depth 4. Unlocks at PRE_OPERATIONAL stage. Risk: model bias compounds geometrically at depth 4 — mitigation via DreamRealityDivergenceTracker.

**Affective Morphing.** A 16-dimensional mood GRU modulates learning rate (0.5x to 2.0x) and exploration noise (0 to 0.3) based on reward history. Curiosity increases exploration; frustration increases learning rate; satisfaction decreases learning rate to preserve good weights. Unlocks at PRE_OPERATIONAL stage.

**Symbiotic Evolution.** Modules share learned representations via weight patches when cosine similarity exceeds 0.7. Enables emergent specialization without explicit supervision. Unlocks at FORMAL stage. Risk: bad abstractions can spread — mitigation via quarantine windows and rollback capability.

### 3.4 Developmental System

**ODE Growth Controller.** Capacity grows according to:

dP/dt = r × P × (1 - P/K)

where P = current parameters, K = 500M (ceiling), r = 0.01 (growth rate).

Growth signal uses the OLG counter-based pattern:
```
signal = smoothed_error + variance_bonus  [ADDITIVE — prevents runaway]

if signal > threshold for trigger_steps consecutive steps:
    SPAWN_LORA_ADAPTER()
    reset counter
    enter cooldown (100 steps)
```

Parameters: threshold=0.1, trigger_steps=10, cooldown=100.

Note: The additive signal formula (error + variance_bonus) rather than multiplicative (error × novelty) was validated through the OLG research program. Multiplicative signals cause either runaway growth or growth starvation depending on the scale of each factor.

**Stage Manager.** Five Piaget-inspired developmental stages, each unlocking different capabilities:

| Stage | Capabilities Unlocked | Transition Requirements |
|---|---|---|
| SENSORIMOTOR | Basic action-outcome learning | Baseline performance |
| PRE_OPERATIONAL | + Dreaming, + Affective Morphing | Min steps + performance threshold |
| CONCRETE | + Introspection | No recent regressions |
| FORMAL | + Symbiotic Evolution | Consistent abstract reasoning |
| METACOGNITIVE | All + Meta-learning | Shadow evaluation pass |

The METACOGNITIVE stage requires a no-regret shadow evaluation loop before unlock — the highest-risk transition, as meta-learning can destabilize prior capabilities.

**LoRA Spawner.** Adds capacity on growth triggers without retraining:
- Rank: 16
- Alpha: 32
- Applied as: output = original_output + lora_adapter(x)
- Low-rank, local, reversible, cheap — correct growth primitive.

### 3.5 Safety Layer

**Safety Priors.** Structural risk assessment with veto probability. Actions with risk > 0.8 trigger safety masking. Critically: safety is a structural filter, not a learned reward signal. This is the direct lesson from the Competence-Conservatism Paradox — learned safety critics suppress risk-taking; structural safety priors do not.

**Productive Conflict.** Domain-aware thresholds derived directly from FAA research:

```python
domain_thresholds = {
    'maze':   0.2,   # Exploration-heavy: allow more risk
    'puzzle': 0.4,   # Logic-heavy: more conservative
    'game':   0.25,  # Balanced
}
```

The system measures goal-action alignment and rewards healthy disagreement between high and low-level modules. This prevents the bureaucratic stagnation identified in the FAA Competence-Conservatism Paradox — a competent safety layer that correctly detects conflict will not suppress risk-taking because the resolution mechanism explicitly tolerates productive conflict.

---

## 4. DIAMOND Integration

### 4.1 DiamondDreamer

DIAMOND replaces abstract noise imagination with photorealistic diffusion-based world model rollouts. The DiamondDreamer wrapper:

1. Receives current observation from HDAGI
2. HDAGI selects action via its policy
3. DIAMOND predicts next state via diffusion sampling
4. Experience tuple (obs, action, reward, next_obs) returned to HDAGI
5. HDAGI learns from imagined experience

### 4.2 Hybrid Training Loop

Training alternates between real environment experience and imagined rollouts:

- Real phase: collect experience, update HDAGI
- Imagination phase: dream_horizon rollouts per real step
- Imagination ratio: 2:1 (imagined:real)
- Growth check: ODE signal computed after each episode

### 4.3 Hardware Profile

| Component | Parameters | VRAM |
|---|---|---|
| HDAGI core | 7.4M | ~0.7 GB |
| DIAMOND world model | 13.5M | ~2-3 GB |
| Activations + gradients | — | ~2 GB |
| **Total** | **20.9M** | **~5-6 GB** |
| RTX 3060 available | — | 12 GB ✅ |

---

## 5. Architecture Validation

Independent validation of the HDAGI architecture confirmed:

✅ Internally consistent — no contradictions between components  
✅ Incorporates lessons from all prior failures (OLG, FAA variance trap, HierHypGen decoder bottleneck)  
✅ Avoids common AGI-architecture mistakes (entangled concerns, always-on powerful mechanisms, learned safety)  
✅ Implementable on consumer hardware  
✅ Scales without "just add more parameters"

**The meta-insight from validation:**

> "Intelligence emerges from constrained growth, not maximized optimization."

HDAGI enforces: delay power, limit authority, encourage disagreement, allow uncertainty, grow only when stuck.

**Identified pressure points requiring ongoing monitoring:**

1. Introspection gradient injection can destabilize during symbiosis → IntrospectionEnergyLimiter
2. Dream nesting hallucination compounds at depth 4 → DreamRealityDivergenceTracker
3. Symbiotic patch infection can spread bad abstractions → quarantine windows + rollback
4. Premature stage transitions → shadow evaluation before METACOGNITIVE unlock

---

## 6. Training Status

**Phase 1 (Complete):** Basic DIAMOND integration, single-game (Breakout), validation of VRAM profile (0.26 GB observed).

**Phase 2 (In Progress):** Mixed-domain training on Breakout + Pong rotation. OLG growth pattern applied. Checkpoints every 10 episodes. First growth event expected within ~10 episodes.

**Phase 3 (Planned):** 100-500 episode long run with growth enabled. Capture first stage transition. Benchmark reward curves.

**Phase 4 (Future):** Custom DIAMOND world model trained on MiniGrid environments. Test stage transitions with custom world model.

---

## 7. Discussion

### 7.1 The Synthesis Principle

HDAGI demonstrates that architectural progress does not require more parameters — it requires synthesizing the right lessons from prior work. Each component of HDAGI directly addresses a failure mode discovered in a prior research program:

- FAA discovered that learned safety suppresses capability → HDAGI uses structural safety
- HTMA identified that attention is a hardware-era constraint → HDAGI uses SSM
- HierHypGen proved structural inductive bias beats statistical learning → HDAGI encodes safety structurally
- DIAMOND provides realistic imagination → HDAGI dreams in photorealistic environments

The architecture that emerges from this synthesis is fundamentally different from architectures designed by adding components to transformers. It is designed around constraints, not capabilities.

### 7.2 Relationship to AGI

HDAGI does not claim to be AGI. It claims to be a defensible architecture for developmental AI that avoids the known failure modes of current approaches. The five Piaget-inspired stages are not a roadmap to AGI — they are a principled way to introduce powerful capabilities only when the system has demonstrated the prerequisites to use them safely.

The METACOGNITIVE stage — where the system can improve its own learning — is the most dangerous transition. The shadow evaluation requirement before unlock is the direct application of the lesson from the Competence-Conservatism Paradox: competent safety is not automatically capability-preserving.

### 7.3 Limitations

**Scale.** Current implementation at 20.9M parameters is a proof-of-concept. The developmental system is designed to grow, but the ceiling of 500M parameters has not been tested.

**World model generalization.** DIAMOND is pretrained on Atari. Custom world models for other domains require dataset collection and retraining.

**Stage transitions.** No stage transitions have been observed yet. Phase 3 training will provide the first empirical data on transition dynamics.

---

## 8. Conclusion

HDAGI represents the convergence of four independent research programs into a unified architecture. The key insight is architectural: intelligence does not emerge from maximized optimization but from constrained, staged growth with structural safety guarantees.

The four prior research programs each contributed one essential element:
- FAA: productive conflict resolution that tolerates disagreement
- HTMA: continuous state evolution replacing stateless attention
- HierHypGen: structural inductive bias replacing statistical learning
- DIAMOND: realistic imagination replacing abstract noise

Together, these elements form an architecture that grows in capability over time, encodes safety as structure rather than penalty, reasons more deeply when uncertain, and dreams in photorealistic imagined environments — all on consumer hardware.

**The research program is ongoing.** Architecture validated. Phase 2 training underway. First stage transition pending.

---

**Code:** https://github.com/Devignite25/AI-Research-Development  
**Acknowledgments.** HDAGI synthesizes lessons from the FAA, HTMA, and HierHypGen research programs conducted at DevIgnite LLC. DIAMOND world model by Alonso et al. (2024).

---

## References

1. Maure, J. A. (2026). The Competence-Conservatism Paradox: Emergent Risk-Aversion in Multi-Agent Safety Architectures. DevIgnite LLC. [This repository]
2. Maure, J. A. (2026). Hierarchical Topological Manifold Architecture (HTMA). DevIgnite LLC. [This repository]
3. Maure, J. A. (2026). HierHypGen: Sheaf-Theoretic Diffusion for Structurally Consistent Image Generation. DevIgnite LLC. [This repository]
4. Alonso, E., et al. (2024). Diffusion for World Modeling: Visual Details Matter in Atari. NeurIPS. (DIAMOND)
5. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
6. Piaget, J. (1952). The Origins of Intelligence in Children. International Universities Press.
7. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
8. Huang et al. (2025). Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable. arXiv:2503.00555.

---

*© 2026 DevIgnite LLC — AI Research & Development / The Wider Lens Research Initiative. Licensed under CC BY 4.0.*
