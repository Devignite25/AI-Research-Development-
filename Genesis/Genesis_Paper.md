# Genesis: A Unified Foundation for Developmental Artificial General Intelligence

**Jorge A. Maure**  
DevIgnite LLC — AI Research & Development  
The Wider Lens Research Initiative  
Jacksonville, Florida, USA  
March 2026

---

## Abstract

We present **Genesis**, a unified architecture for developmental artificial general intelligence that synthesizes six independent DevIgnite research programs spanning December 2025 through March 2026. Genesis integrates: (1) OLG's empirically validated counter-based growth triggering and ODE dynamics; (2) FAA's self-healing pain detection and outcome-based fasting; (3) HTMA's principle of persistent memory over stateless attention; (4) HierHypGen's structural inductive bias principle; (5) HDAGI's five-stage developmental gating and uncertainty-triggered depth; and (6) three novel contributions absent from all prior work — Anticipatory Growth, Objective Divergence, and a Grounded Self-Model.

Genesis starts at 50M parameters and grows to a 1B ceiling via ODE-governed dynamics. The Deliberation Layer contains three structurally distinct agents — Explorer (RNN, novelty objective), Exploiter (FFN, reward objective), and Critic (pain detection + grounded self-model) — resolved by an Arbitrator using weighted voting with Critic veto. A Self-Healing Layer monitors four pain sources (calibration, conflict, gradient, loss) and triggers outcome-based fasting followed by ODE regeneration. The Grounded Self-Model predicts observable environment outcomes rather than internal states, preventing the hallucination compounding identified in HDAGI's introspection system.

Genesis is designed for consumer hardware implementation. The six-phase build plan begins with MiniGrid validation and ends with zero-shot transfer experiments across Atari, MiniGrid, and ARC.

**Keywords:** developmental AGI, anticipatory growth, objective divergence, grounded self-model, self-healing, hierarchical recurrence, multi-objective RL

---

## 1. Introduction

### 1.1 The Research Arc

Genesis is the sixth paper in a continuous research program, not a standalone proposal. Each prior paper solved one critical problem. Genesis is the architecture that emerges when all six solutions are combined.

| Paper | Core Discovery | Genesis Integration |
|---|---|---|
| OLG (Dec 2025) | Counter-based growth triggering works empirically | Growth Controller backbone |
| FAA (Dec 2025) | Competence-Conservatism Paradox + self-healing | Self-Healing Layer + Critic design |
| HTMA (Mar 2026) | Attention is a hardware-era constraint | Mamba SSM throughout |
| HierHypGen (Feb 2026) | Structural bias > statistical learning | Safety encoded structurally |
| HDAGI (Dec 2025–Mar 2026) | First working synthesis, stage-gated | Recurrence Core + Stage Manager |
| Genesis (Mar 2026) | Complete specification + 3 novel contributions | This paper |

### 1.2 What Genesis Adds That No Prior Work Had

**Anticipatory Growth.** Every prior growth system — including OLG — is reactive. Growth fires when the system is already struggling. Genesis introduces a DifficultyPredictor that estimates task difficulty from observation and triggers growth *before* performance degrades. Grow before failure, not after.

**Objective Divergence.** Prior multi-agent safety systems (FAA) used agents with different reward signals but the same architecture. Genesis uses architecturally distinct agents: the Explorer is RNN-based for temporal novelty tracking across episodes; the Exploiter is FFN-based for fast reward optimization within episodes. Different objectives require different computational structures, not just different loss functions.

**Grounded Self-Model.** HDAGI's Recursive Introspection predicts internal states — the system's own hidden representations. This produces useful self-monitoring but is vulnerable to hallucination compounding: errors in the self-model feed back into the self-model. Genesis's Grounded Self-Model predicts *observable environment outcomes* (reward, done, next observation embedding). Grounding to external reality breaks the feedback loop and provides a reliable error signal.

### 1.3 What Genesis Is Not

- ❌ Not a monolithic transformer
- ❌ Not a pure reinforcement learner
- ❌ Not a fixed-capacity model
- ❌ Not a single-objective optimizer
- ❌ Not purely neural — includes explicit structural mechanisms

---

## 2. Architecture

### 2.1 System Overview

```
INPUT FUSION LAYER
(Vision + Text + Audio + Proprioceptive → Unified Embedder, d_model=512)
        │
        ▼
RECURRENCE CORE
├── High-Level Planner (4 Mamba layers, d_state=128, slow)
└── Low-Level Executor (6 Mamba layers, d_state=64, fast)
    └── Uncertainty → triggers deeper recurrence (max depth 8)
        │
        ▼
DELIBERATION LAYER
├── Explorer    (RNN, novelty objective)
├── Exploiter   (FFN, reward objective)
├── Critic      (pain detection + grounded self-model)
└── Arbitrator  (weighted voting + Critic veto)
        │
        ▼
DEVELOPMENTAL LAYER
├── Growth Controller (anticipatory + reactive, ODE-governed)
├── Stem Hypernetwork (error-informed weight generation)
├── Graceful Shrinking (prune unused, merge similar)
└── Stage Manager (SENSORIMOTOR→SYMBOLIC→METACOGNITIVE→TRANSFER)
        │
        ▼
SELF-HEALING LAYER
├── Pain Detector (4 sources: calibration, conflict, gradient, loss)
├── Fasting Manager (outcome-based exit)
├── ODE Regeneration (dS/dt = r·S·(1-S/K) + fasting_boost)
└── Selective Dreaming (importance = TD_error × surprise × recency)
        │
        ▼
OUTPUT LAYER
├── Action Head (discrete + continuous)
├── Value Head (state value + advantage)
└── Meta Head (uncertainty + predicted success + introspection debt)
```

### 2.2 Input Fusion Layer

Unified multimodal embedding at d_model=512:

| Modality | Encoder | Key Parameter |
|---|---|---|
| Vision | Patch encoder | patch_size=16, image_size=224 |
| Text | Tokenizer | vocab_size=32K, max_len=2048 |
| Audio | Mel encoder | mel_bins=80, max_len=3000 |
| Proprioceptive | Linear projection | domain-dependent |

Novel: unified positional encoding across modalities using learned modality tokens. The model learns that position-3 in vision and position-3 in text are structurally analogous, not identical.

### 2.3 Recurrence Core

The HRM backbone from HDAGI, upgraded with expanded d_state and last-step gradient optimization:

```python
RecurrenceCoreConfig:
    d_model = 512
    n_high_layers = 4      # Strategic planning
    n_low_layers = 6       # Reactive execution
    d_state_high = 128     # Larger for abstraction
    d_state_low = 64       # Faster for reactions
    max_recurrence_depth = 8
    uncertainty_threshold = 0.3
    use_last_step_grads = True   # Memory-efficient BPTT
    h_cycles = 4
    l_cycles = 8
```

Bidirectional communication: High sends abstract goals to Low; Low sends feedback and uncertainty signal to High. When uncertainty > 0.3, recurrence depth increases up to max 8 — the system thinks harder when it needs to, not always.

Last-step gradient optimization replaces full BPTT with gradient computation from only the final recurrence step. This provides O(1) memory cost for recurrence training versus O(depth) for full BPTT, with empirically comparable quality.

### 2.4 Deliberation Layer — Objective Divergence

The core novel contribution of Genesis at the architectural level: three agents with genuinely different architectures, not just different objectives.

**Explorer (RNN-based, novelty objective)**

Recurrent architecture for tracking novelty across time. The RNN hidden state accumulates a visitation model — which states have been seen, how recently, how often. Novelty reward is inversely proportional to visitation count. The RNN architecture is necessary: FFN cannot accumulate a temporal visitation model without explicit memory.

```
explorer_type: RNN
hidden_dim: 256
layers: 2
objective: maximize_novelty
```

**Exploiter (FFN-based, reward objective)**

Feedforward architecture for fast within-episode reward maximization. No temporal accumulation needed — the Exploiter maps current state to best-known action. FFN is faster, more stable under reward shaping, and less susceptible to gradient interference from the Explorer's novelty signal.

```
exploiter_type: FFN
hidden_dim: 512
layers: 3
objective: maximize_reward
```

**Critic (oversight + grounded self-model)**

Monitors system health across four pain sources and maintains the Grounded Self-Model. Neither RNN nor FFN — uses a small dedicated network for each pain source, combined into a unified pain signal.

```
pain_sources: [calibration, conflict, gradient, loss]
pain_thresholds: (0.3=stressed, 0.6=inflamed, 0.85=critical)
```

**Arbitrator**

Resolves Explorer/Exploiter conflict using weighted voting based on confidence and history:

```
base weights: Explorer=0.3, Exploiter=0.5, Critic=0.2
exploration_schedule: cosine_decay
arbitrator_mode: weighted_vote with Critic veto
```

Critic veto is absolute: if pain > 0.85 (critical), no action is taken regardless of Explorer/Exploiter agreement. This is the structural safety mechanism — it cannot be overridden by gradient pressure.

### 2.5 Developmental Layer — Anticipatory Growth

**The Novel Mechanism:**

```python
class AnticipatoryGrowthController:
    def should_grow(self, observation, current_capacity):
        predicted_difficulty = self.difficulty_predictor(observation)
        capacity_needed = self.estimate_capacity(predicted_difficulty)

        # Grow preemptively if needed capacity > 80% of current
        if capacity_needed > current_capacity * 0.8:
            return True, "anticipatory"

        # Fall back to OLG-style reactive growth
        return self.reactive_controller.should_grow(...)
```

The DifficultyPredictor is a small network (dim=64) trained jointly with the main system on a curriculum prediction objective. It learns to map observations to expected learning difficulty before the system encounters the task.

**Growth parameters:**
```
initial_params: 50M
max_params: 1B
growth_rate: 0.01
growth_signal: multiplicative (error × novelty × prediction)
trigger_steps: 10
cooldown: 100
```

Note: Genesis uses a *multiplicative* growth signal (error × novelty × prediction), unlike OLG's additive formula (error + variance). The third factor — predicted difficulty — is the key difference. The multiplicative formula is appropriate here because all three factors must be simultaneously elevated to justify growth. A high-error, low-novelty, easy-predicted task does not need capacity growth.

**Stem Hypernetwork:**

Rather than spawning generic LoRA adapters, Genesis uses a Stem Hypernetwork — a small network that generates adapter weights conditioned on the current error signal. Adapters grown under high calibration error have different initial weights than adapters grown under high gradient conflict. Error-informed weight generation accelerates adaptation convergence.

**Graceful Shrinking:**

Adapters with usage < 0.01 for > 10,000 steps are pruned. Adapters with cosine similarity > 0.95 are merged. ODE-governed decay prevents sudden capacity loss. The system can shrink as well as grow — correct for long-horizon training where early-curriculum capacity may become redundant.

**Stage Manager — Four Stages:**

Genesis extends HDAGI's five stages to four more focused stages with hybrid gating (hard-coded veto + learned classifier suggestion):

| Stage | Capabilities | Hard-Coded Gate |
|---|---|---|
| SENSORIMOTOR | Basic perception + action | Baseline reward threshold |
| SYMBOLIC | + Abstract planning, language grounding | No regression in 500 steps |
| METACOGNITIVE | + Self-monitoring, introspection | Shadow eval pass |
| TRANSFER | + Zero-shot generalization, meta-learning | Multi-domain validation |

### 2.6 Self-Healing Layer

**Pain Detection:**

Four pain sources, each with dedicated detector:

```
calibration_pain: prediction_error vs actual_outcome (from Grounded Self-Model)
conflict_pain:    Explorer vs Exploiter disagreement above threshold
gradient_pain:    gradient norm variance — learning instability
loss_pain:        sustained high loss — capacity insufficient
```

Combined via EMA (alpha=0.1) into unified pain level.

**Outcome-Based Fasting:**

Improvement over FAA's time-based fasting:

```python
def fasting_step(self):
    # EXIT if healed — don't wait for timeout
    if current_pain < exit_threshold (0.2):
        self.exit_fasting("healed")
        return

    # Still enforce maximum duration
    if steps >= max_steps (100):
        self.exit_fasting("timeout")
        return

    self.apply_pruning(ratio=0.1)
```

The key improvement: fasting exits when healing is complete, not when a timer expires. This prevents both under-fasting (exiting before healing) and over-fasting (continuing after healing, wasting training time).

**ODE Regeneration:**

```
dS/dt = r × S × (1 - S/K) + fasting_boost_factor
```

The fasting_boost_factor (0.2) accelerates regeneration post-fasting — the system rebounds faster after a completed healing cycle. Stem adapters spawn during regeneration (max 10), providing fresh capacity for the recovered system.

**Selective Dreaming:**

```
importance = td_error × surprise × recency
if importance > threshold (0.5):
    consolidate(memory)
```

Not all memories are worth consolidating. High TD error means the system was surprised by the outcome. High surprise means the observation was unusual. High recency means the memory is fresh. Only memories meeting all three criteria enter the dream buffer — this prevents low-information experience from dominating consolidation.

### 2.7 Grounded Self-Model

The key distinction from HDAGI's Recursive Introspection:

| System | Predicts | Grounding | Failure Mode |
|---|---|---|---|
| HDAGI RIC | Internal hidden states | None — self-referential | Hallucination compounding |
| Genesis GSM | Observable outcomes | External environment | Standard prediction error |

```python
class GroundedSelfModel:
    def forward(self, state, action):
        return {
            "reward":        self.reward_head(state, action),
            "done":          self.done_head(state, action),
            "next_obs_embed": self.dynamics_head(state, action),
        }

    def compute_grounding_error(self, predictions, actuals):
        return {k: F.mse_loss(predictions[k], actuals[k])
                for k in predictions}
```

Grounding error feeds directly into calibration_pain. When the self-model diverges from reality, pain increases, fasting triggers, and the model is corrected. The feedback loop runs through observable reality, not through internal state — this is the architectural guarantee against hallucination.

---

## 3. Training Configuration

```
optimizer: Lion (from FAA research)
learning_rate: 1e-4
weight_decay: 0.01
warmup_steps: 1000
batch_size: 8
accumulation_steps: 4
precision: bf16
gradient_checkpointing: True

Multi-objective weights:
  explorer: 0.3
  exploiter: 0.5
  critic: 0.2
  grounding: 0.1
```

---

## 4. Implementation Phases

### Phase 1: Core Foundation (Weeks 1–2)
- Input Fusion Layer (vision + text)
- Recurrence Core (Mamba backbone, last-step grads)
- Basic action/value heads

**Validation:** MiniGrid-Empty-5x5 reaching >0.9 reward

### Phase 2: Deliberation (Weeks 3–4)
- Explorer (RNN, novelty objective)
- Exploiter (FFN, reward objective)
- Critic (pain detection)
- Arbitrator (weighted voting)

**Validation:** Explorer finds novel states faster than Exploiter alone

### Phase 3: Developmental (Weeks 5–6)
- ODE growth controller
- Stem hypernetwork
- Anticipatory growth predictor
- Graceful shrinking
- Stage manager

**Validation:** System grows from 50M to 100M+ during curriculum

### Phase 4: Self-Healing (Weeks 7–8)
- Extended pain detector
- Outcome-based fasting
- ODE regeneration
- Selective dreaming

**Validation:** Recovery from artificially induced pain within 100 steps

### Phase 5: Grounded Self-Model (Weeks 9–10)
- Observable outcome prediction
- Grounding error computation
- Integration with Critic

**Validation:** Self-model predicts reward with <0.1 MSE

### Phase 6: Integration & Transfer (Weeks 11–12)
- Full system integration
- Multi-domain training (Atari, MiniGrid, ARC)
- Zero-shot transfer experiments

**Validation:** >0% zero-shot transfer to held-out domains

---

## 5. Validation Experiments

**Experiment 1: Recurrence Efficiency**
Does last-step gradient match full BPTT quality with less memory?
Setup: Sudoku-9×9, 1000 samples. Metrics: accuracy, training time, GPU memory.

**Experiment 2: Anticipatory vs Reactive Growth**
Does growing before failure improve learning speed?
Setup: Easy→Hard curriculum. Metrics: steps to solve Hard, total capacity used.

**Experiment 3: Objective Divergence**
Do architecturally distinct agents produce different behaviors than same-architecture agents with different rewards?
Setup: Explorer vs Exploiter on exploration task. Metrics: state coverage, policy divergence.

**Experiment 4: Grounded vs Ungrounded Self-Models**
Does predicting observables improve generalization vs predicting internal states?
Setup: Train domain A, test domain B. Metrics: transfer performance, self-model error.

**Experiment 5: Outcome-Based Healing**
Does outcome-based fasting heal faster than time-based fasting?
Setup: Induce pain artificially, measure recovery. Metrics: steps to healthy, final performance.

---

## 6. Discussion

### 6.1 The Synthesis Principle

Six independent research programs, each addressing one failure mode, converge on a single architecture. The synthesis principle: **architectural progress comes from combining the right lessons, not from adding parameters.**

OLG proved growth can be signal-driven. FAA proved safety must be structural. HTMA proved attention is not fundamental. HierHypGen proved inductive bias beats statistics. HDAGI proved these can be combined and run on consumer hardware. Genesis completes the picture with anticipatory growth, objective divergence, and grounded self-modeling.

### 6.2 Why Anticipatory Growth Matters

Reactive growth is expensive: the system must degrade before it grows. In a curriculum setting, degradation triggers forgetting cascades that require additional recovery steps. Anticipatory growth — if the difficulty predictor is accurate — eliminates the degradation phase entirely. The system arrives at a hard task already equipped.

The difficulty predictor accuracy is the key unknown. Phase 3 validation will measure this directly.

### 6.3 Why Objective Divergence Matters

Using the same architecture for exploration and exploitation creates gradient interference. The Explorer's novelty gradients and the Exploiter's reward gradients conflict at the parameter level in a shared architecture. Separate architectures eliminate this interference at the cost of separate parameter budgets. At Genesis's scale (50M–1B), the cost is acceptable. The benefit — clean gradient separation — is architectural.

### 6.4 Limitations

**Difficulty predictor training.** The AnticipatoryGrowthController requires a DifficultyPredictor trained on curriculum data. Cold-start performance (before the predictor is calibrated) may be worse than reactive growth. A warm-up period using reactive growth only is recommended.

**Multi-objective stability.** Four competing gradient signals (explorer, exploiter, critic, grounding) require careful weight tuning. The weights provided are initial estimates based on FAA research; empirical tuning during Phase 2 is expected.

**World model absent.** Unlike HDAGI, Genesis does not integrate DIAMOND for photorealistic dreaming. Selective dreaming uses replay of real experience only. DIAMOND integration is a natural extension for a post-Phase 6 version.

---

## 7. Conclusion

Genesis is the culmination of six research programs conducted at DevIgnite LLC between December 2025 and March 2026. It begins where OLG established that signal-driven growth works. It incorporates the safety architecture FAA discovered was necessary. It builds on the memory principles HTMA proposed. It applies the structural bias lesson HierHypGen proved. It extends the working synthesis HDAGI demonstrated.

And it adds three things none of the prior work had: a system that grows before it fails, agents that are architecturally distinct by objective, and a self-model grounded in observable reality rather than internal prediction.

The architecture is specified. The build plan is defined. The validation experiments are designed.

> *"Intelligence is not a capacity you start with. It is a structure you grow into."*

---

**Code:** https://github.com/Devignite25/AI-Research-Development  
**Acknowledgments.** Genesis synthesizes all prior DevIgnite research programs. The complete lineage is documented in the companion papers in this repository.

---

## References

1. Maure, J. A. (2025). OLG: Ontogenetic Layer Genesis. DevIgnite LLC. [This repository]
2. Maure, J. A. (2025). The FAA Architecture and the Competence-Conservatism Paradox. DevIgnite LLC. [This repository]
3. Maure, J. A. (2026). HTMA: Hierarchical Topological Manifold Architecture. DevIgnite LLC. [This repository]
4. Maure, J. A. (2026). HierHypGen: Sheaf-Theoretic Diffusion. DevIgnite LLC. [This repository]
5. Maure, J. A. (2026). HDAGI: Hierarchical Developmental AGI. DevIgnite LLC. [This repository]
6. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
7. Alonso, E., et al. (2024). Diffusion for World Modeling: Visual Details Matter in Atari. NeurIPS. (DIAMOND)
8. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
9. Piaget, J. (1952). The Origins of Intelligence in Children. International Universities Press.
10. Chen, X., et al. (2023). Lion: Symbolic Discovery of Optimization Algorithms. arXiv:2302.06675.

---

*© 2026 DevIgnite LLC — AI Research & Development / The Wider Lens Research Initiative. Licensed under CC BY 4.0.*
